# Mycelium GCP VM Deployment

Simple deployment: Single VM + SQLite + Vertex AI APIs

## Architecture

```
┌─────────────────────────────────────┐
│         GCP VM (e2-small)           │
│  ┌─────────────┐  ┌──────────────┐  │
│  │   Python    │  │   SQLite     │  │
│  │  Mycelium   │──│  mycelium.db │  │
│  └──────┬──────┘  └──────────────┘  │
└─────────┼───────────────────────────┘
          │ Vertex AI APIs
          ▼
    ┌─────────────┐
    │  Vertex AI  │
    │  - Gemini   │
    │  - Embeddings│
    └─────────────┘
```

## Cost

- **e2-small**: ~$13/month (2 vCPU, 2GB RAM)
- **e2-medium**: ~$25/month (2 vCPU, 4GB RAM)
- **Vertex AI**: Pay per use (~$0.0001/embedding, ~$0.0003/1K LLM tokens)

## Quick Start

```bash
# Set your project
export GCP_PROJECT_ID="your-project-id"

# Create VM
./setup.sh create

# Deploy code
./setup.sh deploy

# SSH in and run
./setup.sh ssh

# On the VM:
cd /opt/mycelium
source venv/bin/activate
python -m mycelium.pipeline_runner --dataset gsm8k --limit 10
```

## Commands

| Command | Description |
|---------|-------------|
| `./setup.sh create` | Create new VM |
| `./setup.sh deploy` | Deploy/update code |
| `./setup.sh ssh` | SSH into VM |
| `./setup.sh run <args>` | Run mycelium command |
| `./setup.sh delete` | Delete VM |
| `./setup.sh status` | Show VM status |

## Configuration

Environment variables (set before running):

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | mycelium-prod2 | GCP project |
| `GCP_ZONE` | us-central1-a | VM zone |
| `VM_NAME` | mycelium-vm | VM instance name |
| `MACHINE_TYPE` | e2-small | VM size |

## Running as a Service

To run mycelium as a background service:

```bash
# On the VM, create systemd service
sudo tee /etc/systemd/system/mycelium.service << 'EOF'
[Unit]
Description=Mycelium Math Solver
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/mycelium
Environment=PATH=/opt/mycelium/venv/bin:/usr/bin
ExecStart=/opt/mycelium/venv/bin/python -m mycelium.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable mycelium
sudo systemctl start mycelium

# Check status
sudo systemctl status mycelium
```

## Backup Database

```bash
# From your local machine
gcloud compute scp mycelium-vm:/opt/mycelium/mycelium.db ./backup-$(date +%Y%m%d).db \
    --zone=us-central1-a
```

## Upgrade VM Size

```bash
# Stop VM
gcloud compute instances stop mycelium-vm --zone=us-central1-a

# Change machine type
gcloud compute instances set-machine-type mycelium-vm \
    --machine-type=e2-medium \
    --zone=us-central1-a

# Start VM
gcloud compute instances start mycelium-vm --zone=us-central1-a
```
