#!/usr/bin/env python3
"""
Deploy Phase 2 data generation Lambda function.

Creates or updates the Lambda function with:
- 3GB memory (NOT 1GB!)
- 15 minute timeout
- Python 3.11 runtime

Usage:
    python lambda/phase2_datagen/deploy.py --create    # First time
    python lambda/phase2_datagen/deploy.py --update    # Update code
"""

import os
import sys
import zipfile
import argparse
import tempfile
from pathlib import Path

import boto3


FUNCTION_NAME = "phase2-datagen"
RUNTIME = "python3.11"
MEMORY_MB = 3072  # 3GB - IMPORTANT: not 1GB!
TIMEOUT_SEC = 900  # 15 minutes
HANDLER = "handler.lambda_handler"


def create_deployment_package() -> bytes:
    """Create zip file with Lambda code."""
    script_dir = Path(__file__).parent

    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add handler
            handler_path = script_dir / "handler.py"
            zf.write(handler_path, "handler.py")

        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def get_or_create_role(iam) -> str:
    """Get or create IAM role for Lambda."""
    role_name = "phase2-datagen-role"

    try:
        response = iam.get_role(RoleName=role_name)
        print(f"Using existing role: {role_name}")
        return response['Role']['Arn']
    except iam.exceptions.NoSuchEntityException:
        pass

    print(f"Creating role: {role_name}")

    # Trust policy
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=str(trust_policy).replace("'", '"'),
        Description="Role for Phase 2 data generation Lambda",
    )
    role_arn = response['Role']['Arn']

    # Attach policies
    policies = [
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    ]

    for policy_arn in policies:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        print(f"  Attached: {policy_arn.split('/')[-1]}")

    # Wait for role to propagate
    import time
    print("Waiting for role to propagate...")
    time.sleep(10)

    return role_arn


def create_function(lambda_client, iam, zip_bytes: bytes):
    """Create new Lambda function."""
    role_arn = get_or_create_role(iam)

    print(f"\nCreating Lambda function: {FUNCTION_NAME}")
    print(f"  Runtime: {RUNTIME}")
    print(f"  Memory: {MEMORY_MB} MB")
    print(f"  Timeout: {TIMEOUT_SEC} sec")

    response = lambda_client.create_function(
        FunctionName=FUNCTION_NAME,
        Runtime=RUNTIME,
        Role=role_arn,
        Handler=HANDLER,
        Code={'ZipFile': zip_bytes},
        Description="Phase 2 data generation for Mycelium",
        Timeout=TIMEOUT_SEC,
        MemorySize=MEMORY_MB,
        Publish=True,
    )

    print(f"\nCreated: {response['FunctionArn']}")
    return response


def update_function(lambda_client, zip_bytes: bytes):
    """Update existing Lambda function code."""
    print(f"Updating Lambda function: {FUNCTION_NAME}")

    response = lambda_client.update_function_code(
        FunctionName=FUNCTION_NAME,
        ZipFile=zip_bytes,
        Publish=True,
    )

    # Also update configuration
    lambda_client.update_function_configuration(
        FunctionName=FUNCTION_NAME,
        MemorySize=MEMORY_MB,
        Timeout=TIMEOUT_SEC,
    )

    print(f"Updated: {response['FunctionArn']}")
    print(f"  Memory: {MEMORY_MB} MB")
    print(f"  Timeout: {TIMEOUT_SEC} sec")
    return response


def main():
    parser = argparse.ArgumentParser(description="Deploy Phase 2 Lambda")
    parser.add_argument("--create", action="store_true", help="Create new function")
    parser.add_argument("--update", action="store_true", help="Update existing function")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")

    args = parser.parse_args()

    if not args.create and not args.update:
        print("Specify --create or --update")
        sys.exit(1)

    # Create clients
    lambda_client = boto3.client('lambda', region_name=args.region)
    iam = boto3.client('iam', region_name=args.region)

    # Create deployment package
    print("Creating deployment package...")
    zip_bytes = create_deployment_package()
    print(f"Package size: {len(zip_bytes) / 1024:.1f} KB")

    # Deploy
    if args.create:
        create_function(lambda_client, iam, zip_bytes)
    else:
        update_function(lambda_client, zip_bytes)

    print("\nDone!")


if __name__ == "__main__":
    main()
