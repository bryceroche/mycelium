# Mycelium GCP Deployment
# All-in-one container for Cloud Run with Vertex AI + Cloud SQL
#
# Build: docker build -f deploy/gcp/Dockerfile -t mycelium-gcp .
# Run locally: docker run -e MYCELIUM_PROVIDER=local mycelium-gcp
# Deploy to GCP: see deploy/gcp/README.md

FROM python:3.11-slim

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MYCELIUM_PROVIDER=gcp

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .
COPY requirements.txt .
COPY deploy/gcp/requirements-gcp.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-gcp.txt

# Copy source code and pyproject
COPY src/ src/
COPY pyproject.toml .

# Create empty README.md (required by pyproject.toml)
RUN touch README.md

# Install the package (not editable for container)
RUN pip install --no-cache-dir .

# Create non-root user for security
RUN useradd -m -u 1000 mycelium
USER mycelium

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the HTTP server for Cloud Run
CMD ["python", "-m", "mycelium.server"]
