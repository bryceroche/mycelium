# Mycelium GCP Deployment

Deploy Mycelium code to GCP, using Google's managed services:

- **Google Cloud SQL** - PostgreSQL database (replaces local SQLite)
- **Google Vertex AI Gemini** - LLM for decomposition/generation (replaces OpenAI)
- **Google Vertex AI Embeddings** - text-embedding-004, 768d (matches MathBERT, pgvector compatible)
- **Google Cloud Run** - Runs the Mycelium code

No data migration needed - the system learns fresh on GCP with Google's models.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GCP Project                          │
│                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │  Cloud Run   │────▶│  Vertex AI   │     │  Cloud SQL  │ │
│  │  (Mycelium)  │     │  (Gemini +   │     │ (PostgreSQL │ │
│  │              │     │  Embeddings) │     │ + pgvector) │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│         │                                         ▲         │
│         │         VPC Private Network            │         │
│         └────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. GCP Project with billing enabled
2. `gcloud` CLI installed and authenticated
3. Terraform installed (for infrastructure)
4. Docker installed (for building images)

## Quick Start

### 1. Set up environment

```bash
export PROJECT_ID="your-gcp-project"
export REGION="us-central1"

gcloud config set project $PROJECT_ID
```

### 2. Deploy infrastructure with Terraform

```bash
cd deploy/gcp/terraform

# Initialize Terraform
terraform init

# Create a strong password for the database
DB_PASSWORD=$(openssl rand -base64 32)

# Plan and apply
terraform plan -var="project_id=$PROJECT_ID" -var="db_password=$DB_PASSWORD"
terraform apply -var="project_id=$PROJECT_ID" -var="db_password=$DB_PASSWORD"
```

### 3. Initialize the database schema

```bash
# Get Cloud SQL connection name from Terraform output
CONNECTION_NAME=$(terraform output -raw cloud_sql_connection_name)

# Use Cloud SQL Auth Proxy for local connection
cloud-sql-proxy $CONNECTION_NAME &

# Run schema initialization
PGPASSWORD=$DB_PASSWORD psql -h 127.0.0.1 -U mycelium -d mycelium -f ../init_schema.sql
```

### 4. Build and push Docker image

```bash
cd ../../..  # Back to project root

# Build image
docker build -f deploy/gcp/Dockerfile -t gcr.io/$PROJECT_ID/mycelium:latest .

# Push to Container Registry
docker push gcr.io/$PROJECT_ID/mycelium:latest
```

### 5. Deploy to Cloud Run

```bash
# Update Cloud Run service with new image
gcloud run services update mycelium \
  --region=$REGION \
  --image=gcr.io/$PROJECT_ID/mycelium:latest
```

## Environment Variables

The following environment variables are set automatically by Terraform:

| Variable | Description |
|----------|-------------|
| `MYCELIUM_PROVIDER` | Set to `gcp` |
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_REGION` | Deployment region |
| `CLOUD_SQL_CONNECTION_NAME` | Cloud SQL instance connection string |
| `CLOUD_SQL_DB_NAME` | Database name (mycelium) |
| `CLOUD_SQL_USER` | Database user |
| `CLOUD_SQL_PASSWORD` | Database password (from Secret Manager) |

## Local Development with GCP Backend

You can run locally while using GCP services:

```bash
# Start Cloud SQL Auth Proxy
cloud-sql-proxy $CONNECTION_NAME &

# Set environment variables
export MYCELIUM_PROVIDER=gcp
export GCP_PROJECT_ID=$PROJECT_ID
export GCP_REGION=$REGION
export CLOUD_SQL_CONNECTION_NAME=$CONNECTION_NAME
export CLOUD_SQL_DB_NAME=mycelium
export CLOUD_SQL_USER=mycelium
export CLOUD_SQL_PASSWORD=$DB_PASSWORD

# Run locally
python -m mycelium
```

## Vertex AI Models

### LLM (Gemini)
Model automatically selected based on `TRAINING_MODE`:

| Mode | Model | Why |
|------|-------|-----|
| Training (`TRAINING_MODE=true`) | `gemini-3.0-flash` | Better decompositions = better signatures |
| Inference (`TRAINING_MODE=false`) | `gemini-1.5-flash` | Cheaper, but rarely used (zero-LLM routing) |

Override: `export VERTEX_AI_MODEL_TRAINING=gemini-1.5-pro`

### Embeddings
- Default: `text-embedding-004` (768 dimensions)
- Matches MathBERT dimensions and stays within pgvector's 2000-dim limit

## Scaling

### Cloud Run
- Min instances: 0 (scale to zero)
- Max instances: 10 (configurable in Terraform)
- CPU: 2 vCPU
- Memory: 4GB

### Cloud SQL
- Default tier: `db-f1-micro` (development)
- Production: Consider `db-custom-2-4096` or higher
- Update in `terraform/main.tf`: `variable "db_tier"`

## Costs

Estimated monthly costs (us-central1, minimal usage):

| Service | Estimate |
|---------|----------|
| Cloud SQL (db-f1-micro) | ~$10/month |
| Cloud Run | Pay per use (~$0.00002/request) |
| Vertex AI Gemini | ~$0.00025/1K input tokens |
| Vertex AI Embeddings | ~$0.00002/1K tokens |

For training runs, costs scale with usage. Monitor in GCP Console.

## Monitoring

```bash
# View Cloud Run logs
gcloud run services logs read mycelium --region=$REGION

# View Cloud SQL metrics
gcloud sql instances describe mycelium-db
```

## Cleanup

```bash
cd deploy/gcp/terraform

# Disable deletion protection first
terraform apply -var="project_id=$PROJECT_ID" -var="db_password=$DB_PASSWORD" \
  -target=google_sql_database_instance.postgres \
  -var='deletion_protection=false'

# Destroy all resources
terraform destroy -var="project_id=$PROJECT_ID" -var="db_password=$DB_PASSWORD"
```
