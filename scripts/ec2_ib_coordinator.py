#!/usr/bin/env python3
"""
EC2 Coordinator for MapReduce IB Template Discovery

Orchestrates the distributed processing of 33GB IAF data across EC2 workers.

Usage:
  # Start workers and run full pipeline
  python ec2_ib_coordinator.py run --workers 4

  # Check status of running workers
  python ec2_ib_coordinator.py status

  # Collect results and run reduce
  python ec2_ib_coordinator.py reduce

  # Cleanup (stop all workers)
  python ec2_ib_coordinator.py cleanup
"""

import argparse
import boto3
import json
import subprocess
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Dict


# ============================================================================
# CONFIGURATION
# ============================================================================
S3_BUCKET = "mycelium-data"
S3_IAF_PREFIX = "iaf_extraction"
S3_RESULTS_PREFIX = "ib_results"

# Shard layout (from your S3 listing)
SHARDS = {
    'instance1': ['gpu0', 'gpu1', 'gpu2', 'gpu3', 'gpu4', 'gpu5', 'gpu6', 'gpu7'],
    'instance2': ['gpu0', 'gpu1', 'gpu2', 'gpu3', 'gpu4', 'gpu5', 'gpu6', 'gpu7'],
}

# EC2 worker configuration
WORKER_INSTANCE_IDS = [
    "i-0bc407b5be4d8d236",  # mycelium-train
    "i-0292641b5a234f74e",  # mycelium-worker (4xlarge)
    "i-09d02720e937aa869",  # mycelium-worker
    "i-0b52f20fd5cd3e4dd",  # mycelium-worker
    "i-04b83dac2f11ce2b2",  # mycelium-worker
]

# SSH key
SSH_KEY = "~/.ssh/mycelium-key.pem"
SSH_USER = "ubuntu"


@dataclass
class WorkerAssignment:
    """Assignment of shards to a worker."""
    instance_id: str
    public_ip: Optional[str]
    shards: List[str]  # e.g., ["instance1/gpu0", "instance1/gpu1"]


# ============================================================================
# EC2 HELPERS
# ============================================================================
def get_ec2_client():
    return boto3.client('ec2')


def start_instances(instance_ids: List[str]) -> List[str]:
    """Start EC2 instances and wait for them to be running."""
    ec2 = get_ec2_client()

    print(f"Starting {len(instance_ids)} instances...")
    ec2.start_instances(InstanceIds=instance_ids)

    print("Waiting for instances to be running...")
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=instance_ids)

    # Get public IPs
    response = ec2.describe_instances(InstanceIds=instance_ids)
    ips = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            ip = instance.get('PublicIpAddress', 'N/A')
            name = next((t['Value'] for t in instance.get('Tags', []) if t['Key'] == 'Name'), 'Unknown')
            print(f"  {instance['InstanceId']} ({name}): {ip}")
            ips.append(ip)

    # Wait for SSH to be available
    print("Waiting for SSH to be available...")
    time.sleep(30)

    return ips


def stop_instances(instance_ids: List[str]):
    """Stop EC2 instances."""
    ec2 = get_ec2_client()
    print(f"Stopping {len(instance_ids)} instances...")
    ec2.stop_instances(InstanceIds=instance_ids)
    print("Instances stopping...")


def get_instance_ips(instance_ids: List[str]) -> Dict[str, str]:
    """Get public IPs of running instances."""
    ec2 = get_ec2_client()
    response = ec2.describe_instances(InstanceIds=instance_ids)

    ips = {}
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            if instance['State']['Name'] == 'running':
                ips[instance['InstanceId']] = instance.get('PublicIpAddress')

    return ips


# ============================================================================
# SSH HELPERS
# ============================================================================
def ssh_command(ip: str, cmd: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Execute SSH command on remote host."""
    ssh_cmd = [
        'ssh', '-i', Path(SSH_KEY).expanduser().as_posix(),
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'ConnectTimeout=10',
        f'{SSH_USER}@{ip}',
        cmd
    ]
    return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)


def scp_to_remote(ip: str, local_path: str, remote_path: str):
    """Copy file to remote host."""
    scp_cmd = [
        'scp', '-i', Path(SSH_KEY).expanduser().as_posix(),
        '-o', 'StrictHostKeyChecking=no',
        local_path,
        f'{SSH_USER}@{ip}:{remote_path}'
    ]
    return subprocess.run(scp_cmd, capture_output=True, text=True)


def scp_from_remote(ip: str, remote_path: str, local_path: str):
    """Copy file from remote host."""
    scp_cmd = [
        'scp', '-r', '-i', Path(SSH_KEY).expanduser().as_posix(),
        '-o', 'StrictHostKeyChecking=no',
        f'{SSH_USER}@{ip}:{remote_path}',
        local_path
    ]
    return subprocess.run(scp_cmd, capture_output=True, text=True)


# ============================================================================
# WORKER COORDINATION
# ============================================================================
def assign_shards_to_workers(num_workers: int) -> List[WorkerAssignment]:
    """Distribute shards evenly across workers."""
    # Flatten shard list
    all_shards = []
    for instance, gpus in SHARDS.items():
        for gpu in gpus:
            all_shards.append(f"{instance}/{gpu}")

    # Distribute to workers
    assignments = []
    shards_per_worker = len(all_shards) // num_workers
    extra = len(all_shards) % num_workers

    idx = 0
    for i in range(num_workers):
        n_shards = shards_per_worker + (1 if i < extra else 0)
        worker_shards = all_shards[idx:idx + n_shards]
        idx += n_shards

        assignments.append(WorkerAssignment(
            instance_id=WORKER_INSTANCE_IDS[i] if i < len(WORKER_INSTANCE_IDS) else None,
            public_ip=None,
            shards=worker_shards,
        ))

    return assignments


def setup_worker(ip: str) -> bool:
    """Setup worker environment."""
    print(f"  Setting up worker at {ip}...")

    # Install dependencies
    setup_cmd = """
    cd ~ &&
    source venv/bin/activate 2>/dev/null || python3 -m venv venv &&
    source venv/bin/activate &&
    pip install -q ijson scikit-learn numpy scipy boto3 &&
    mkdir -p ~/ib_results
    """

    result = ssh_command(ip, setup_cmd, timeout=120)
    if result.returncode != 0:
        print(f"    ERROR: Setup failed: {result.stderr}")
        return False

    # Copy the script
    script_path = Path(__file__).parent / "mapreduce_ib_discovery.py"
    scp_result = scp_to_remote(ip, str(script_path), "~/mapreduce_ib_discovery.py")
    if scp_result.returncode != 0:
        print(f"    ERROR: Failed to copy script: {scp_result.stderr}")
        return False

    print(f"    Worker {ip} ready")
    return True


def run_map_on_worker(ip: str, shards: List[str]) -> bool:
    """Run MAP phase on a worker."""
    # Build shard paths
    shard_args = []
    for shard in shards:
        instance, gpu = shard.split('/')
        s3_path = f"s3://{S3_BUCKET}/{S3_IAF_PREFIX}/{instance}/iaf_v3_{gpu}_valid.json"
        shard_args.append(s3_path)

    shard_paths = ",".join(shard_args)

    cmd = f"""
    cd ~ &&
    source venv/bin/activate &&
    python mapreduce_ib_discovery.py map --shard-paths '{shard_paths}' --output-dir ~/ib_results 2>&1 |
    tee ~/ib_map.log
    """

    print(f"  Starting MAP on {ip} with {len(shards)} shards...")
    result = ssh_command(ip, cmd, timeout=3600)  # 1 hour timeout

    if result.returncode != 0:
        print(f"    ERROR on {ip}: {result.stderr[:500]}")
        return False

    print(f"    MAP complete on {ip}")
    return True


def collect_results(ips: List[str], local_dir: Path) -> bool:
    """Collect MAP results from all workers."""
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting results to {local_dir}...")

    for i, ip in enumerate(ips):
        print(f"  Collecting from {ip}...")
        worker_dir = local_dir / f"worker_{i}"
        worker_dir.mkdir(exist_ok=True)

        result = scp_from_remote(ip, "~/ib_results/*", str(worker_dir))
        if result.returncode != 0:
            print(f"    WARNING: Failed to collect from {ip}: {result.stderr}")

    # Merge all worker results into single directory
    merged_dir = local_dir / "merged"
    merged_dir.mkdir(exist_ok=True)

    for worker_dir in local_dir.glob("worker_*"):
        for f in worker_dir.glob("*.json"):
            dest = merged_dir / f.name
            if not dest.exists():
                import shutil
                shutil.copy(f, dest)

    print(f"  Results merged to {merged_dir}")
    return True


def upload_results_to_s3(local_dir: Path):
    """Upload final results to S3."""
    s3_dest = f"s3://{S3_BUCKET}/{S3_RESULTS_PREFIX}/"

    cmd = ['aws', 's3', 'sync', str(local_dir), s3_dest]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Results uploaded to {s3_dest}")
    else:
        print(f"ERROR uploading to S3: {result.stderr}")


# ============================================================================
# MAIN COMMANDS
# ============================================================================
def cmd_run(args):
    """Run the full MapReduce pipeline."""
    num_workers = args.workers

    print("=" * 70)
    print("MAPREDUCE IB TEMPLATE DISCOVERY")
    print("=" * 70)
    print(f"Workers: {num_workers}")
    print(f"Total shards: {sum(len(v) for v in SHARDS.values())}")

    # Assign shards to workers
    assignments = assign_shards_to_workers(num_workers)
    print("\nShard assignments:")
    for i, a in enumerate(assignments):
        print(f"  Worker {i}: {len(a.shards)} shards - {a.shards}")

    # Start instances
    instance_ids = [a.instance_id for a in assignments if a.instance_id]
    ips = start_instances(instance_ids)

    # Update assignments with IPs
    for i, ip in enumerate(ips):
        assignments[i].public_ip = ip

    # Setup workers in parallel
    print("\nSetting up workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(setup_worker, a.public_ip): a for a in assignments}
        for future in as_completed(futures):
            assignment = futures[future]
            try:
                success = future.result()
                if not success:
                    print(f"  WARNING: Worker {assignment.public_ip} setup failed")
            except Exception as e:
                print(f"  ERROR: Worker {assignment.public_ip} setup exception: {e}")

    # Run MAP phase in parallel
    print("\n" + "=" * 70)
    print("PHASE 1: MAP")
    print("=" * 70)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_map_on_worker, a.public_ip, a.shards): a for a in assignments}
        for future in as_completed(futures):
            assignment = futures[future]
            try:
                success = future.result()
            except Exception as e:
                print(f"  ERROR on worker {assignment.public_ip}: {e}")

    map_time = time.time() - start_time
    print(f"\nMAP phase complete in {map_time:.1f}s")

    # Collect results
    print("\n" + "=" * 70)
    print("PHASE 2: COLLECT")
    print("=" * 70)

    local_results = Path("/tmp/ib_mapreduce_results")
    collect_results([a.public_ip for a in assignments], local_results)

    # Run REDUCE locally
    print("\n" + "=" * 70)
    print("PHASE 3: REDUCE")
    print("=" * 70)

    reduce_cmd = [
        sys.executable,
        str(Path(__file__).parent / "mapreduce_ib_discovery.py"),
        "reduce",
        "--results-dir", str(local_results / "merged"),
        "--output-dir", str(local_results / "final"),
    ]
    subprocess.run(reduce_cmd)

    # Upload to S3
    if args.upload:
        upload_results_to_s3(local_results / "final")

    # Cleanup
    if args.cleanup:
        print("\nStopping workers...")
        stop_instances(instance_ids)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Results: {local_results / 'final'}")


def cmd_status(args):
    """Check status of workers."""
    ips = get_instance_ips(WORKER_INSTANCE_IDS)

    print("Worker status:")
    for iid, ip in ips.items():
        if ip:
            result = ssh_command(ip, "tail -5 ~/ib_map.log 2>/dev/null || echo 'No log yet'", timeout=10)
            print(f"\n  {iid} ({ip}):")
            print(f"    {result.stdout[:200]}")
        else:
            print(f"\n  {iid}: Not running")


def cmd_reduce(args):
    """Run REDUCE phase on collected results."""
    local_results = Path(args.results_dir)

    reduce_cmd = [
        sys.executable,
        str(Path(__file__).parent / "mapreduce_ib_discovery.py"),
        "reduce",
        "--results-dir", str(local_results / "merged"),
        "--output-dir", str(local_results / "final"),
    ]
    subprocess.run(reduce_cmd)


def cmd_cleanup(args):
    """Stop all workers."""
    stop_instances(WORKER_INSTANCE_IDS)


def main():
    parser = argparse.ArgumentParser(description='EC2 Coordinator for IB Template Discovery')
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # run command
    run_parser = subparsers.add_parser('run', help='Run full pipeline')
    run_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    run_parser.add_argument('--upload', action='store_true', help='Upload results to S3')
    run_parser.add_argument('--cleanup', action='store_true', help='Stop workers after completion')

    # status command
    subparsers.add_parser('status', help='Check worker status')

    # reduce command
    reduce_parser = subparsers.add_parser('reduce', help='Run REDUCE on collected results')
    reduce_parser.add_argument('--results-dir', type=str, default='/tmp/ib_mapreduce_results')

    # cleanup command
    subparsers.add_parser('cleanup', help='Stop all workers')

    args = parser.parse_args()

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'reduce':
        cmd_reduce(args)
    elif args.command == 'cleanup':
        cmd_cleanup(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
