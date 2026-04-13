#!/usr/bin/env bash
# =============================================================================
# build-and-deploy.sh
#
# Complete setup script for Parakeet TDT 0.6b v3
# on NVIDIA DGX Spark (ARM64, CUDA 13, Blackwell GB10)
#
# Run this script on your DGX Spark.
# =============================================================================

set -euo pipefail

# ── Colors for output ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# =============================================================================
# STEP 0: Verify we're on DGX Spark
# =============================================================================
echo ""
echo "============================================================"
echo " Parakeet TDT v3 — DGX Spark Deployment"
echo "============================================================"
echo ""

info "Checking system..."

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    warn "Expected aarch64, got $ARCH. This script is designed for DGX Spark ARM64."
    read -p "Continue anyway? [y/N] " -n 1 -r; echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# Check NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found. Is the NVIDIA driver installed?"
fi

info "GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Check Docker
if ! command -v docker &>/dev/null; then
    fail "Docker not found. Install Docker first."
fi

# Check NVIDIA Container Toolkit
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    warn "NVIDIA Container Toolkit may not be configured."
    warn "Ensure 'nvidia' runtime is available in Docker."
fi

# Check CUDA version
if command -v nvcc &>/dev/null; then
    info "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"
fi

ok "System checks passed"

# =============================================================================
# STEP 1: Set up project directory
# =============================================================================
echo ""
info "Setting up project directory..."

PROJECT_DIR="$HOME/Docker/parakeet/parakeet-spark"

cd "$PROJECT_DIR"
ok "Project directory: $PROJECT_DIR"

# =============================================================================
# STEP 2: Create the Docker network (if not exists)
# =============================================================================
echo ""
info "Ensuring Docker network 'dgx_net' exists..."

if ! docker network inspect dgx_net &>/dev/null; then
    docker network create dgx_net
    ok "Created network 'dgx_net'"
else
    ok "Network 'dgx_net' already exists"
fi

# =============================================================================
# STEP 3: Log in to NGC
# =============================================================================
echo ""
info "Logging in to NVIDIA NGC registry..."

if [[ -z "${NGC_API_KEY:-}" ]]; then
    echo ""
    echo "  You need an NGC API key to pull NVIDIA containers."
    echo "  Get one at: https://org.ngc.nvidia.com/setup/api-key"
    echo ""
    read -sp "  Enter your NGC API key: " NGC_API_KEY
    echo ""
fi

echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
ok "Logged in to nvcr.io"

# Save for docker-compose
export NGC_API_KEY

# =============================================================================
# STEP 4: Pull the base image
# =============================================================================
echo ""
info "Pulling NVIDIA PyTorch base image (ARM64)..."
info "This is ~15 GB and may take a while on first pull."
echo ""

docker pull nvcr.io/nvidia/pytorch:25.11-py3
ok "Base image pulled"

# =============================================================================
# STEP 5: Build the Parakeet ASR container
# =============================================================================
echo ""
info "Building Parakeet TDT v3 container..."
info "This will:"
info "  1. Install NeMo toolkit + dependencies"
info "  2. Download the Parakeet TDT 0.6b v3 model (~1.2 GB)"
info "  3. Set up the FastAPI server"
info ""
info "Expected build time: 15-30 minutes (first time)"
echo ""

docker build \
    --tag parakeet-tdt-v3-spark:latest \
    --progress=plain \
    --shm-size=8g \
    -f docker/Dockerfile \
    .

ok "Parakeet ASR container built: parakeet-tdt-v3-spark:latest"

# =============================================================================
# STEP 6: Quick smoke test of ASR container
# =============================================================================
echo ""
info "Running smoke test..."

# Start container briefly to verify GPU access and model loading
SMOKE_ID=$(docker run -d --rm \
    --gpus all \
    --shm-size=8g \
    --name parakeet-smoke-test \
    -e CUDA_VISIBLE_DEVICES=0 \
    parakeet-tdt-v3-spark:latest \
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('Smoke test passed!')
")

# Wait for it to finish
docker wait "$SMOKE_ID" 2>/dev/null || true

# Show output
docker logs parakeet-smoke-test 2>/dev/null || true

ok "Smoke test complete"

# =============================================================================
# STEP 7: Create .env file
# =============================================================================
echo ""
info "Creating .env file..."

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    cat > "$PROJECT_DIR/.env" <<EOF
NGC_API_KEY=${NGC_API_KEY}
EOF
    chmod 600 "$PROJECT_DIR/.env"
    ok "Created .env with your NGC API key"
else
    ok ".env already exists"
fi

# =============================================================================
# STEP 8: Deploy with docker compose
# =============================================================================
echo ""
echo "============================================================"
echo " Deployment Options"
echo "============================================================"
echo ""
echo "  A) Deploy now with docker compose (recommended for testing)"
echo "  B) I'll deploy via Portainer (show me the steps)"
echo "  C) Skip — I'll deploy later"
echo ""
read -p "  Choose [A/B/C]: " -n 1 -r DEPLOY_CHOICE; echo

case $DEPLOY_CHOICE in
    [Aa])
        echo ""
        info "Starting services with docker compose..."
        # Updated to point to docker/ folder
        docker compose -f docker/docker-compose.yml --env-file .env up -d
        echo ""
        ok "Services starting!"
        info "Watch logs with:"
        info "  docker compose -f docker/docker-compose.yml logs -f parakeet-asr"
        ;;
    [Bb])
        echo ""
        echo "============================================================"
        echo " Portainer Deployment Steps"
        echo "============================================================"
        echo ""
        echo "  1. Open Portainer: https://<your-spark-ip>:9443"
        echo ""
        echo "  2. Go to: Stacks → Add Stack"
        echo ""
        echo "  3. Name: voice-agent"
        echo ""
        echo "  4. Build method: Upload"
        # Updated to point to docker/ folder
        echo "     Upload: $PROJECT_DIR/docker/portainer-stack.yml"
        echo ""
        echo "  5. Environment variables → Add:"
        echo "     NGC_API_KEY = $NGC_API_KEY"
        echo ""
        echo "  6. Click 'Deploy the stack'"
        echo ""
        echo "  NOTE: The parakeet-asr image was already built locally."
        echo "        Portainer will find it by the image name."
        echo ""
        ;;
    *)
        info "Skipping deployment."
        ;;
esac

# =============================================================================
# STEP 9: Print summary
# =============================================================================
echo ""
echo "============================================================"
echo " Setup Complete!"
echo "============================================================"
echo ""
echo "  Built images:"
echo "    • parakeet-tdt-v3-spark:latest (Parakeet ASR)"
echo ""
echo "  Endpoints (after deployment):"
echo "    ASR Transcription:  http://<spark-ip>:8010/v1/audio/transcriptions"
echo "    ASR Health:         http://<spark-ip>:8010/health"
echo "    ASR Info:           http://<spark-ip>:8010/"
echo ""
echo "  Test ASR:"
echo '    curl -s http://localhost:8010/v1/audio/transcriptions \'
echo '      -F file="@test.wav" -F language=auto | python3 -m json.tool'
echo ""
echo "  Test with German audio:"
echo '    curl -s http://localhost:8010/v1/audio/transcriptions \'
echo '      -F file="@german.wav" -F language=de | python3 -m json.tool'
echo ""
echo "  Supported languages: bg, cs, da, de, el, en, es, et, fi, fr,"
echo "    hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk"
echo ""
echo "  Project dir: $PROJECT_DIR"
echo "  Logs:        docker compose -f docker/docker-compose.yml logs -f"
echo ""
echo "============================================================"
