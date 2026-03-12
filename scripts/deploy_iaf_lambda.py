#!/usr/bin/env python3
"""
Deploy IAF Operand Extraction Lambda Function.

Sets up:
- Lambda function with 3GB memory, 300s timeout
- S3 trigger on s3://mycelium-data/iaf_extraction/chunked/
- IAM role with S3 read/write permissions

Usage:
    python scripts/deploy_iaf_lambda.py --create
    python scripts/deploy_iaf_lambda.py --update
    python scripts/deploy_iaf_lambda.py --trigger
    python scripts/deploy_iaf_lambda.py --invoke-all
"""

import argparse
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

FUNCTION_NAME = "mycelium-iaf-operand-extraction"
REGION = "us-east-1"
MEMORY_MB = 3072  # 3GB
TIMEOUT_S = 300   # 5 minutes
RUNTIME = "python3.11"

SOURCE_BUCKET = "mycelium-data"
SOURCE_PREFIX = "iaf_extraction/chunked/"
OUTPUT_BUCKET = "mycelium-data-v7"
OUTPUT_PREFIX = "iaf_operands/"

ROLE_NAME = "mycelium-lambda-s3-role"


def create_deployment_package():
    """Create ZIP deployment package."""
    project_root = Path(__file__).parent.parent
    plan_dir = project_root / "plan"

    # Create temp zip
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Main handler
        iaf_script = plan_dir / "iaf_operand_extraction.py"
        zf.write(iaf_script, "iaf_operand_extraction.py")

        # Lambda wrapper
        wrapper = '''
import json
from iaf_operand_extraction import lambda_handler as handler

def lambda_handler(event, context):
    return handler(event, context)
'''
        zf.writestr("lambda_function.py", wrapper)

    print(f"Created deployment package: {zip_path}")
    return zip_path


def create_iam_role():
    """Create IAM role for Lambda."""
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    # Check if role exists
    result = subprocess.run(
        ["aws", "iam", "get-role", "--role-name", ROLE_NAME],
        capture_output=True
    )

    if result.returncode == 0:
        print(f"Role {ROLE_NAME} already exists")
        role_data = json.loads(result.stdout)
        return role_data["Role"]["Arn"]

    # Create role
    result = subprocess.run([
        "aws", "iam", "create-role",
        "--role-name", ROLE_NAME,
        "--assume-role-policy-document", json.dumps(trust_policy)
    ], capture_output=True, check=True)

    role_arn = json.loads(result.stdout)["Role"]["Arn"]
    print(f"Created role: {role_arn}")

    # Attach S3 full access policy
    subprocess.run([
        "aws", "iam", "attach-role-policy",
        "--role-name", ROLE_NAME,
        "--policy-arn", "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    ], check=True)

    # Attach CloudWatch logs policy
    subprocess.run([
        "aws", "iam", "attach-role-policy",
        "--role-name", ROLE_NAME,
        "--policy-arn", "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    ], check=True)

    print("Attached S3 and CloudWatch policies")

    # Wait for role to propagate
    import time
    time.sleep(10)

    return role_arn


def get_role_arn():
    """Get existing role ARN."""
    result = subprocess.run(
        ["aws", "iam", "get-role", "--role-name", ROLE_NAME],
        capture_output=True
    )
    if result.returncode == 0:
        return json.loads(result.stdout)["Role"]["Arn"]
    return None


def create_lambda():
    """Create Lambda function."""
    role_arn = create_iam_role()
    zip_path = create_deployment_package()

    try:
        result = subprocess.run([
            "aws", "lambda", "create-function",
            "--function-name", FUNCTION_NAME,
            "--runtime", RUNTIME,
            "--role", role_arn,
            "--handler", "lambda_function.lambda_handler",
            "--zip-file", f"fileb://{zip_path}",
            "--memory-size", str(MEMORY_MB),
            "--timeout", str(TIMEOUT_S),
            "--region", REGION
        ], capture_output=True, check=True)

        print(f"Created Lambda function: {FUNCTION_NAME}")
        print(f"  Memory: {MEMORY_MB}MB")
        print(f"  Timeout: {TIMEOUT_S}s")

    finally:
        os.unlink(zip_path)


def update_lambda():
    """Update Lambda function code."""
    zip_path = create_deployment_package()

    try:
        subprocess.run([
            "aws", "lambda", "update-function-code",
            "--function-name", FUNCTION_NAME,
            "--zip-file", f"fileb://{zip_path}",
            "--region", REGION
        ], capture_output=True, check=True)

        print(f"Updated Lambda function: {FUNCTION_NAME}")

    finally:
        os.unlink(zip_path)


def setup_s3_trigger():
    """Set up S3 trigger for Lambda."""
    # Get Lambda ARN
    result = subprocess.run([
        "aws", "lambda", "get-function",
        "--function-name", FUNCTION_NAME,
        "--region", REGION
    ], capture_output=True, check=True)

    lambda_arn = json.loads(result.stdout)["Configuration"]["FunctionArn"]

    # Add permission for S3 to invoke Lambda
    try:
        subprocess.run([
            "aws", "lambda", "add-permission",
            "--function-name", FUNCTION_NAME,
            "--statement-id", f"s3-{SOURCE_BUCKET}-trigger",
            "--action", "lambda:InvokeFunction",
            "--principal", "s3.amazonaws.com",
            "--source-arn", f"arn:aws:s3:::{SOURCE_BUCKET}",
            "--region", REGION
        ], capture_output=True, check=True)
        print("Added S3 invoke permission")
    except subprocess.CalledProcessError:
        print("S3 permission already exists")

    # Configure S3 notification
    notification_config = {
        "LambdaFunctionConfigurations": [{
            "LambdaFunctionArn": lambda_arn,
            "Events": ["s3:ObjectCreated:*"],
            "Filter": {
                "Key": {
                    "FilterRules": [{
                        "Name": "prefix",
                        "Value": SOURCE_PREFIX
                    }]
                }
            }
        }]
    }

    subprocess.run([
        "aws", "s3api", "put-bucket-notification-configuration",
        "--bucket", SOURCE_BUCKET,
        "--notification-configuration", json.dumps(notification_config)
    ], check=True)

    print(f"Set up S3 trigger: s3://{SOURCE_BUCKET}/{SOURCE_PREFIX}*")


def list_chunks():
    """List all IAF chunks."""
    result = subprocess.run([
        "aws", "s3", "ls",
        f"s3://{SOURCE_BUCKET}/{SOURCE_PREFIX}",
        "--recursive"
    ], capture_output=True, text=True)

    chunks = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            parts = line.split()
            if len(parts) >= 4:
                key = parts[-1]
                if key.endswith(".json") or key.endswith(".jsonl"):
                    chunks.append(key)

    return chunks


def invoke_all():
    """Manually invoke Lambda for all chunks."""
    chunks = list_chunks()
    print(f"Found {len(chunks)} chunks to process")

    total_examples = 0
    failed = []

    for i, key in enumerate(chunks):
        print(f"\n[{i+1}/{len(chunks)}] Processing {key}...")

        # Create S3 event payload
        event = {
            "Records": [{
                "s3": {
                    "bucket": {"name": SOURCE_BUCKET},
                    "object": {"key": key}
                }
            }]
        }

        with open("/tmp/lambda_payload.json", "w") as f:
            json.dump(event, f)

        result = subprocess.run([
            "aws", "lambda", "invoke",
            "--function-name", FUNCTION_NAME,
            "--cli-binary-format", "raw-in-base64-out",
            "--payload", "file:///tmp/lambda_payload.json",
            "--region", REGION,
            "/tmp/lambda_response.json"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            try:
                with open("/tmp/lambda_response.json") as f:
                    response = json.load(f)
                examples = response.get("training_examples", 0)
                total_examples += examples
                print(f"  {examples} examples (total: {total_examples})")
            except:
                print(f"  Could not parse response")
                failed.append(key)
        else:
            print(f"  Error: {result.stderr}")
            failed.append(key)

    print(f"\n=== Summary ===")
    print(f"Total examples: {total_examples}")
    print(f"Failed chunks: {len(failed)}")
    if failed:
        print(f"Failed: {failed[:5]}...")


def invoke_single(chunk_key: str):
    """Invoke Lambda for a single chunk."""
    import base64

    event = {
        "Records": [{
            "s3": {
                "bucket": {"name": SOURCE_BUCKET},
                "object": {"key": chunk_key}
            }
        }]
    }

    print(f"Invoking for {chunk_key}...")

    # Write payload to temp file to avoid shell escaping issues
    with open("/tmp/lambda_payload.json", "w") as f:
        json.dump(event, f)

    result = subprocess.run([
        "aws", "lambda", "invoke",
        "--function-name", FUNCTION_NAME,
        "--cli-binary-format", "raw-in-base64-out",
        "--payload", "file:///tmp/lambda_payload.json",
        "--region", REGION,
        "/tmp/lambda_response.json"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        with open("/tmp/lambda_response.json") as f:
            response = json.load(f)
        print(f"Result: {response}")
        return response
    else:
        print(f"Error: {result.stderr}")
        return None


def check_outputs():
    """Check processed output files."""
    result = subprocess.run([
        "aws", "s3", "ls",
        f"s3://{SOURCE_BUCKET}/iaf_operands/",
        "--recursive"
    ], capture_output=True, text=True)

    files = []
    total_size = 0
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            parts = line.split()
            if len(parts) >= 4:
                size = int(parts[2])
                key = parts[-1]
                files.append((key, size))
                total_size += size

    print(f"Processed files: {len(files)}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")

    return files


def invoke_parallel(batch_size: int = 10):
    """Invoke Lambda for multiple chunks in parallel."""
    import concurrent.futures
    import time

    chunks = list_chunks()
    print(f"Found {len(chunks)} chunks to process")

    def invoke_chunk(key):
        event = {
            "Records": [{
                "s3": {
                    "bucket": {"name": SOURCE_BUCKET},
                    "object": {"key": key}
                }
            }]
        }

        payload_file = f"/tmp/lambda_payload_{hash(key)}.json"
        response_file = f"/tmp/lambda_response_{hash(key)}.json"

        with open(payload_file, "w") as f:
            json.dump(event, f)

        result = subprocess.run([
            "aws", "lambda", "invoke",
            "--function-name", FUNCTION_NAME,
            "--cli-binary-format", "raw-in-base64-out",
            "--payload", f"file://{payload_file}",
            "--region", REGION,
            response_file
        ], capture_output=True, text=True)

        if result.returncode == 0:
            try:
                with open(response_file) as f:
                    response = json.load(f)
                return key, response.get("training_examples", 0), None
            except:
                return key, 0, "parse error"
        return key, 0, result.stderr

    total_examples = 0
    failed = []

    # Process in batches
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        print(f"\nBatch {batch_start // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}: {len(batch)} chunks")

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(invoke_chunk, key): key for key in batch}
            for future in concurrent.futures.as_completed(futures):
                key, examples, error = future.result()
                if error:
                    print(f"  {key.split('/')[-1]}: ERROR - {error}")
                    failed.append(key)
                else:
                    total_examples += examples
                    print(f"  {key.split('/')[-1]}: {examples} examples")

    print(f"\n=== Summary ===")
    print(f"Total examples: {total_examples}")
    print(f"Failed chunks: {len(failed)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Create Lambda function")
    parser.add_argument("--update", action="store_true", help="Update Lambda code")
    parser.add_argument("--trigger", action="store_true", help="Set up S3 trigger")
    parser.add_argument("--invoke-all", action="store_true", help="Process all chunks sequentially")
    parser.add_argument("--invoke-parallel", type=int, metavar="N", help="Process all chunks with N parallel workers")
    parser.add_argument("--invoke", type=str, help="Process single chunk by key")
    parser.add_argument("--list", action="store_true", help="List IAF chunks")
    parser.add_argument("--check", action="store_true", help="Check output files")

    args = parser.parse_args()

    if args.create:
        create_lambda()
    elif args.update:
        update_lambda()
    elif args.trigger:
        setup_s3_trigger()
    elif args.invoke_all:
        invoke_all()
    elif args.invoke_parallel:
        invoke_parallel(batch_size=args.invoke_parallel)
    elif args.invoke:
        invoke_single(args.invoke)
    elif args.list:
        chunks = list_chunks()
        print(f"Found {len(chunks)} chunks:")
        for c in chunks[:10]:
            print(f"  {c}")
        if len(chunks) > 10:
            print(f"  ... and {len(chunks) - 10} more")
    elif args.check:
        check_outputs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
