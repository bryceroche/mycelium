# Mycelium GCP Infrastructure
#
# This Terraform configuration creates:
# - Cloud SQL PostgreSQL instance with pgvector
# - Cloud Run service for Mycelium
# - IAM bindings for Vertex AI access
# - VPC connector for private Cloud SQL access
#
# Usage:
#   cd deploy/gcp/terraform
#   terraform init
#   terraform plan -var="project_id=YOUR_PROJECT"
#   terraform apply -var="project_id=YOUR_PROJECT"

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "db_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-f1-micro"  # Start small, scale up as needed
}

variable "db_password" {
  description = "Database password for mycelium user"
  type        = string
  sensitive   = true
}

# Provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "services" {
  for_each = toset([
    "sqladmin.googleapis.com",
    "run.googleapis.com",
    "aiplatform.googleapis.com",
    "vpcaccess.googleapis.com",
    "compute.googleapis.com",
    "servicenetworking.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# VPC for private connectivity
resource "google_compute_network" "vpc" {
  name                    = "mycelium-vpc"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.services]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "mycelium-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
}

# VPC Connector for Cloud Run -> Cloud SQL
resource "google_vpc_access_connector" "connector" {
  name          = "mycelium-connector"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.vpc.name
  depends_on    = [google_project_service.services]
}

# Private Service Connection for Cloud SQL
resource "google_compute_global_address" "private_ip_range" {
  name          = "mycelium-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_range.name]

  depends_on = [google_project_service.services]
}

# Cloud SQL PostgreSQL instance
resource "google_sql_database_instance" "postgres" {
  name             = "mycelium-db"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier = var.db_tier


    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
    }

    backup_configuration {
      enabled = true
    }
  }

  deletion_protection = true
  depends_on          = [google_service_networking_connection.private_vpc_connection]
}

# Database
resource "google_sql_database" "mycelium" {
  name     = "mycelium"
  instance = google_sql_database_instance.postgres.name
}

# Database user
resource "google_sql_user" "mycelium" {
  name     = "mycelium"
  instance = google_sql_database_instance.postgres.name
  password = var.db_password
}

# Service Account for Cloud Run
resource "google_service_account" "mycelium" {
  account_id   = "mycelium-runner"
  display_name = "Mycelium Cloud Run Service Account"
}

# IAM: Vertex AI access
resource "google_project_iam_member" "vertex_ai" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.mycelium.email}"
}

# IAM: Cloud SQL access
resource "google_project_iam_member" "cloudsql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.mycelium.email}"
}

# Cloud Run service
resource "google_cloud_run_v2_service" "mycelium" {
  name     = "mycelium"
  location = var.region

  template {
    service_account = google_service_account.mycelium.email

    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    containers {
      image = "gcr.io/${var.project_id}/mycelium:latest"

      env {
        name  = "MYCELIUM_PROVIDER"
        value = "gcp"
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCP_REGION"
        value = var.region
      }
      env {
        name  = "CLOUD_SQL_CONNECTION_NAME"
        value = google_sql_database_instance.postgres.connection_name
      }
      env {
        name  = "CLOUD_SQL_DB_NAME"
        value = google_sql_database.mycelium.name
      }
      env {
        name  = "CLOUD_SQL_USER"
        value = google_sql_user.mycelium.name
      }
      env {
        name = "CLOUD_SQL_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_password.secret_id
            version = "latest"
          }
        }
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }
  }

  depends_on = [
    google_project_service.services,
    google_sql_database_instance.postgres,
  ]
}

# Secret Manager for DB password
resource "google_secret_manager_secret" "db_password" {
  secret_id = "mycelium-db-password"

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

resource "google_secret_manager_secret_iam_member" "db_password_access" {
  secret_id = google_secret_manager_secret.db_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.mycelium.email}"
}

# Outputs
output "cloud_run_url" {
  value       = google_cloud_run_v2_service.mycelium.uri
  description = "Cloud Run service URL"
}

output "cloud_sql_connection_name" {
  value       = google_sql_database_instance.postgres.connection_name
  description = "Cloud SQL connection name for local testing"
}

output "service_account_email" {
  value       = google_service_account.mycelium.email
  description = "Service account email"
}
