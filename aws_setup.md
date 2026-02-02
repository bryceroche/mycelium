# AWS g5.xlarge Setup for Mycelium

GPU: NVIDIA A10G (24GB VRAM) - runs DeepSeek-Math-7B easily
Cost: ~$1.01/hr on-demand, ~$0.40/hr spot

## 1. Install AWS CLI (one-time)

```bash
# macOS
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
rm AWSCLIV2.pkg

# Verify
aws --version
```

## 2. Configure AWS credentials

```bash
aws configure
# Enter:
#   AWS Access Key ID: (from AWS Console > IAM > Users > Security credentials)
#   AWS Secret Access Key: (same place)
#   Default region: us-east-1
#   Default output format: json
```

## 3. Create the VM

```bash
# Create key pair (one-time)
aws ec2 create-key-pair --key-name mycelium-key --query 'KeyMaterial' --output text > ~/.ssh/mycelium-key.pem
chmod 400 ~/.ssh/mycelium-key.pem

# Create security group (one-time)
aws ec2 create-security-group --group-name mycelium-sg --description "Mycelium GPU VM"
aws ec2 authorize-security-group-ingress --group-name mycelium-sg --protocol tcp --port 22 --cidr 0.0.0.0/0

# Launch g5.xlarge with Deep Learning AMI
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type g5.xlarge \
    --key-name mycelium-key \
    --security-groups mycelium-sg \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mycelium-gpu}]' \
    --query 'Instances[0].InstanceId' \
    --output text

# Note: ami-0c7217cdde317cfec is Ubuntu 22.04 in us-east-1
# For Deep Learning AMI, use: ami-0e001c9271cf7f3b9 (may vary by region)
```

## 4. Get instance IP and SSH

```bash
# Get public IP
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=mycelium-gpu" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text

# SSH in
ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP_ADDRESS>
```

## 5. Setup on VM (first time)

```bash
# Install NVIDIA drivers (if using base Ubuntu AMI)
sudo apt update && sudo apt install -y nvidia-driver-535

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone repo
git clone https://github.com/bryceroche/mycelium.git
cd mycelium

# Install deps
uv sync

# Test GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

## 6. Run attention extraction

```bash
cd ~/mycelium
uv run python -c "
from mycelium.attention import extract_attention
result = extract_attention('John has half the apples that Mary has')
print(f'Tokens: {result.tokens}')
print(f'Attention shape: {result.attention.shape}')
"
```

## 7. Stop/Start/Terminate

```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=mycelium-gpu" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

# Stop (keeps disk, ~$8/mo for 100GB)
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Start again
aws ec2 start-instances --instance-ids $INSTANCE_ID

# Terminate (delete everything)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

## Cost Summary

| State | Cost |
|-------|------|
| Running | ~$1.01/hr |
| Stopped | ~$0.01/hr (disk only) |
| Terminated | $0 |

## Quick Reference

```bash
# SSH shortcut (add to ~/.zshrc)
alias mycelium-ssh='ssh -i ~/.ssh/mycelium-key.pem ubuntu@$(aws ec2 describe-instances --filters "Name=tag:Name,Values=mycelium-gpu" "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].PublicIpAddress" --output text)'
```
