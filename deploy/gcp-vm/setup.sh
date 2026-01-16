#!/bin/bash
# Mycelium GCP VM Setup Script
# Simple deployment: VM + SQLite + Vertex AI APIs
#
# Usage:
#   ./setup.sh create    # Create VM and deploy
#   ./setup.sh deploy    # Deploy code to existing VM
#   ./setup.sh ssh       # SSH into VM
#   ./setup.sh delete    # Delete VM

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-mycelium-prod2}"
ZONE="${GCP_ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-mycelium-vm}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-small}"  # ~$13/month, 2 vCPU, 2GB RAM

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[mycelium]${NC} $1"; }
warn() { echo -e "${YELLOW}[mycelium]${NC} $1"; }
error() { echo -e "${RED}[mycelium]${NC} $1"; exit 1; }

# Get the repo root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

create_vm() {
    log "Creating VM: $VM_NAME in $ZONE..."

    # Create VM with startup script
    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=debian-12 \
        --image-project=debian-cloud \
        --boot-disk-size=20GB \
        --boot-disk-type=pd-balanced \
        --scopes=cloud-platform \
        --tags=mycelium-server \
        --metadata=startup-script='#!/bin/bash
# Install dependencies
apt-get update
apt-get install -y python3-pip python3-venv git

# Create app directory
mkdir -p /opt/mycelium
chown -R $(whoami):$(whoami) /opt/mycelium
'

    log "VM created! Waiting for startup script..."
    sleep 30

    # Create firewall rule for HTTP (optional, for API access)
    gcloud compute firewall-rules create allow-mycelium-http \
        --project="$PROJECT_ID" \
        --allow=tcp:8080 \
        --target-tags=mycelium-server \
        --description="Allow HTTP traffic to Mycelium" \
        2>/dev/null || warn "Firewall rule already exists"

    log "VM ready! Run './setup.sh deploy' to deploy code"
}

deploy_code() {
    log "Deploying code to $VM_NAME..."

    # Sync code to VM (excluding large files)
    gcloud compute scp --recurse \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --compress \
        "$REPO_ROOT/src" \
        "$REPO_ROOT/pyproject.toml" \
        "$VM_NAME:/opt/mycelium/"

    # Create requirements and setup script on VM
    gcloud compute ssh "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command='
cd /opt/mycelium

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies (minimal - no ML models needed)
pip install --upgrade pip
pip install httpx numpy pydantic python-dotenv sympy google-cloud-aiplatform

# Create empty README for pip install
touch README.md

# Install mycelium package
pip install -e .

# Create .env file for Vertex AI
cat > .env << EOF
MYCELIUM_PROVIDER=gcp
GCP_PROJECT_ID='"$PROJECT_ID"'
GCP_REGION=us-central1
TRAINING_MODE=true
EOF

echo "Deployment complete!"
'

    log "Code deployed! Run './setup.sh ssh' to access the VM"
}

ssh_vm() {
    log "Connecting to $VM_NAME..."
    gcloud compute ssh "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE"
}

run_mycelium() {
    log "Running mycelium on $VM_NAME..."
    gcloud compute ssh "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command='
cd /opt/mycelium
source venv/bin/activate
python -m mycelium "$@"
' -- "$@"
}

delete_vm() {
    warn "Deleting VM: $VM_NAME..."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud compute instances delete "$VM_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --quiet
        log "VM deleted"
    fi
}

status() {
    log "VM Status:"
    gcloud compute instances describe "$VM_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --format="table(name,status,networkInterfaces[0].accessConfigs[0].natIP,machineType.basename())" \
        2>/dev/null || warn "VM not found"
}

# Main
case "${1:-}" in
    create)
        create_vm
        ;;
    deploy)
        deploy_code
        ;;
    ssh)
        ssh_vm
        ;;
    run)
        shift
        run_mycelium "$@"
        ;;
    delete)
        delete_vm
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {create|deploy|ssh|run|delete|status}"
        echo ""
        echo "Commands:"
        echo "  create  - Create new VM"
        echo "  deploy  - Deploy/update code on VM"
        echo "  ssh     - SSH into VM"
        echo "  run     - Run mycelium command on VM"
        echo "  delete  - Delete VM"
        echo "  status  - Show VM status"
        exit 1
        ;;
esac
