#!/usr/bin/env python3
"""
MapReduce-style chunking script for large IAF JSON files from S3.

Splits 1-2GB IAF JSON files into smaller chunks for easier processing.
Each file contains a JSON array of records (~15-20MB per record).
Default chunk size: 50 records (~750MB-1GB per chunk, well under S3 limits).

Usage:
  # Default: 50 records per chunk
  python chunk_medusa_files.py

  # Custom chunk size
  python chunk_medusa_files.py --chunk-size 100

  # Dry run (show what would be created)
  python chunk_medusa_files.py --dry-run

  # Process specific files
  python chunk_medusa_files.py --input s3://mycelium-data/iaf_extraction/instance1/iaf_v3_gpu0_valid.json

  # Upload to custom S3 location
  python chunk_medusa_files.py --output s3://my-bucket/custom/path/
"""

import json
import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import glob


def download_from_s3(s3_path: str, local_path: str) -> bool:
    """Download file from S3 using aws s3 cp."""
    print(f"  Downloading {s3_path}...")
    cmd = ["aws", "s3", "cp", s3_path, local_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr}")
        return False

    # Get file size for logging
    file_size = os.path.getsize(local_path)
    size_gb = file_size / (1024**3)
    print(f"    Downloaded {size_gb:.2f}GB")
    return True


def upload_to_s3(local_path: str, s3_path: str) -> bool:
    """Upload file to S3 using aws s3 cp."""
    cmd = ["aws", "s3", "cp", local_path, s3_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ERROR uploading to S3: {result.stderr}")
        return False

    return True


def list_s3_files(s3_pattern: str) -> List[str]:
    """List all files matching S3 pattern using aws s3 ls with filtering."""
    import fnmatch
    
    # Extract bucket and directory prefix (without wildcard)
    path_without_s3 = s3_pattern.replace("s3://", "")
    parts = path_without_s3.split("/")
    bucket = parts[0]
    
    # Get directory prefix (before wildcard part)
    prefix_parts = []
    for part in parts[1:]:
        if "*" in part or "?" in part:
            break
        prefix_parts.append(part)
    
    prefix = "/".join(prefix_parts) + "/" if prefix_parts else ""
    
    # List all files in directory
    cmd = ["aws", "s3", "ls", f"s3://{bucket}/{prefix}", "--recursive"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR listing S3: {result.stderr}")
        return []
    
    # Parse and filter results
    files = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            parts_list = line.split()
            if len(parts_list) >= 4:
                file_path = " ".join(parts_list[3:])
                full_path = f"s3://{bucket}/{file_path}"
                # Match against pattern
                if fnmatch.fnmatch(full_path, s3_pattern):
                    files.append(full_path)
    
    return files


def load_json_array(path: str) -> List[Dict]:
    """Load JSON array from file."""
    print(f"  Loading JSON from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    print(f"  Loaded {len(data)} records")
    return data


def chunk_records(records: List[Dict], chunk_size: int) -> List[List[Dict]]:
    """Split records into chunks."""
    chunks = []
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def save_chunk(chunk: List[Dict], output_path: str) -> bool:
    """Save chunk to local file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(chunk, f)
        return True
    except Exception as e:
        print(f"    ERROR saving chunk: {e}")
        return False


def process_file(s3_path: str, chunk_size: int, output_s3_prefix: str,
                 dry_run: bool = False, keep_local: bool = False) -> int:
    """
    Process a single IAF file: download, chunk, upload.

    Args:
        s3_path: Full S3 path to file
        chunk_size: Number of records per chunk
        output_s3_prefix: S3 prefix for output chunks
        dry_run: If True, just show what would be created
        keep_local: If True, keep local files (don't delete)

    Returns:
        Number of chunks created
    """
    # Extract base filename (e.g., "iaf_v3_gpu0_valid" from full path)
    filename = s3_path.split("/")[-1]
    base_name = filename.replace(".json", "")

    print(f"\nProcessing: {filename}")
    print(f"  S3 path: {s3_path}")

    # Download from S3
    local_temp = f"/tmp/{filename}"
    if not download_from_s3(s3_path, local_temp):
        return 0

    # Load records
    try:
        records = load_json_array(local_temp)
    except Exception as e:
        print(f"  ERROR loading JSON: {e}")
        if os.path.exists(local_temp):
            os.remove(local_temp)
        return 0

    # Split into chunks
    chunks = chunk_records(records, chunk_size)
    print(f"  Created {len(chunks)} chunks of {chunk_size} records each")

    # Save and upload chunks
    local_chunk_dir = f"/tmp/chunks_{base_name}"
    os.makedirs(local_chunk_dir, exist_ok=True)

    chunk_count = 0
    for chunk_idx, chunk in enumerate(chunks):
        chunk_filename = f"{base_name}_chunk_{chunk_idx:03d}.json"
        local_chunk_path = os.path.join(local_chunk_dir, chunk_filename)
        s3_chunk_path = output_s3_prefix.rstrip("/") + "/" + chunk_filename

        # Get chunk size in MB
        chunk_json_str = json.dumps(chunk)
        chunk_size_mb = len(chunk_json_str.encode()) / (1024**2)

        if dry_run:
            print(f"    [{chunk_idx:03d}] {chunk_filename} ({len(chunk)} records, "
                  f"{chunk_size_mb:.1f}MB) -> {s3_chunk_path}")
        else:
            # Save locally
            if not save_chunk(chunk, local_chunk_path):
                continue

            # Upload to S3
            print(f"    [{chunk_idx:03d}] Uploading {chunk_filename} "
                  f"({len(chunk)} records, {chunk_size_mb:.1f}MB)...", end=" ")
            if upload_to_s3(local_chunk_path, s3_chunk_path):
                print("OK")
                chunk_count += 1
            else:
                print("FAILED")

        # Clean up local chunk
        if os.path.exists(local_chunk_path):
            os.remove(local_chunk_path)

    # Clean up temp directory and input file
    if os.path.exists(local_chunk_dir):
        os.rmdir(local_chunk_dir)
    if not keep_local and os.path.exists(local_temp):
        os.remove(local_temp)

    return chunk_count


def main():
    parser = argparse.ArgumentParser(
        description='Chunk large IAF JSON files from S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--chunk-size', type=int, default=50,
        help='Number of records per chunk (default: 50, ~750MB-1GB each)'
    )
    parser.add_argument(
        '--input', type=str,
        help='Input S3 pattern (default: s3://mycelium-data/iaf_extraction/instance1/iaf_v3_gpu*_valid.json)',
        default='s3://mycelium-data/iaf_extraction/instance1/iaf_v3_gpu*_valid.json'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output S3 prefix (default: s3://mycelium-data/iaf_extraction/chunked/)',
        default='s3://mycelium-data/iaf_extraction/chunked/'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be created without uploading'
    )
    parser.add_argument(
        '--keep-local', action='store_true',
        help='Keep local temporary files (for debugging)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MEDUSA IAF FILE CHUNKING")
    print("=" * 70)
    print(f"Input pattern:  {args.input}")
    print(f"Output prefix:  {args.output}")
    print(f"Chunk size:     {args.chunk_size} records")
    print(f"Dry run:        {args.dry_run}")
    print()

    # Find matching files
    print("Finding input files...")
    input_files = list_s3_files(args.input)

    if not input_files:
        print("ERROR: No files found matching pattern")
        sys.exit(1)

    print(f"Found {len(input_files)} files:")
    for f in sorted(input_files):
        print(f"  - {f.split('/')[-1]}")
    print()

    # Process each file
    total_chunks = 0
    for i, s3_path in enumerate(sorted(input_files), 1):
        try:
            chunks = process_file(
                s3_path,
                args.chunk_size,
                args.output,
                dry_run=args.dry_run,
                keep_local=args.keep_local
            )
            total_chunks += chunks
        except Exception as e:
            print(f"ERROR processing file: {e}")
            continue

    # Summary
    print()
    print("=" * 70)
    if args.dry_run:
        print(f"DRY RUN: Would create {total_chunks} chunks")
    else:
        print(f"COMPLETE: Created {total_chunks} chunks")
        print(f"Output location: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
