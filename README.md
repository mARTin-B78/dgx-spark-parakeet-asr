# Parakeet TDT 0.6b v3 — DGX Spark Voice Agent

Multilingual Speech-to-Text (ASR) running on NVIDIA DGX Spark (ARM64 / CUDA 13 / Blackwell GB10) in Docker.

## What You Get

| Service | Model | Languages | Port | API |
|---------|-------|-----------|------|-----|
| **ASR** | Parakeet TDT 0.6b v3 | 25 European | 8010 | OpenAI-compatible |

ASR supported languages: Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish, Ukrainian

---

## Prerequisites

- NVIDIA DGX Spark with Ubuntu (ARM64 / aarch64)
- Docker + NVIDIA Container Toolkit installed
- NGC API key ([get one here](https://org.ngc.nvidia.com/setup/api-key))
- ~20 GB disk space for images + models

---

## Deployment: Two Ways to Install

You can either use the **Pre-built Image** (fastest, recommended for most users) or **Build from Source** (best if you want to modify the code).

### Path A: The Easy Way (Pre-built Docker Image)
*Use this if you just want to run the API and transcribe audio. No code required.*

**Step 1: Set up your folder and API key**
```bash
mkdir -p ~/parakeet-asr && cd ~/parakeet-asr

# Replace with your actual NGC API key
echo "NGC_API_KEY=nvapi-YOUR-KEY-HERE" > .env
chmod 600 .env
```

**Step 2: Create the Docker Compose file**
Save the following as `docker-compose.yml` in that folder:
```yaml
version: '3.8'
services:
  parakeet-asr:
    image: martinb78/parakeet-tdt-v3-spark:latest
    container_name: parakeet-asr
    restart: unless-stopped
    environment:
      - NGC_API_KEY=${NGC_API_KEY}
      - PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3
      - MAX_SEGMENT_SECONDS=1200
      - MAX_UPLOAD_MB=200
      - HF_HOME=/cache/huggingface
      - TRANSFORMERS_CACHE=/cache/huggingface
      - CUDA_VISIBLE_DEVICES=0
      - NCCL_P2P_DISABLE=1
    ports:
      - "8010:8000"
    volumes:
      - parakeet-model-cache:/cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    shm_size: 16gb

volumes:
  parakeet-model-cache:
```

**Step 3: Start the server**
```bash
docker compose up -d
```
*(Docker will download the pre-built image and start the server. The model takes about 60-90 seconds to load into VRAM).*

---

### Path B: Build from Source (For Developers)
*Use this if you want to edit the FastAPI Python code or NeMo settings.*

**Step 1: Clone the repository**
```bash
git clone https://github.com/mARTin-B78/dgx-spark-parakeet-asr.git
cd dgx-spark-parakeet-asr
```

**Step 2: Log in to NVIDIA NGC**
```bash
export NGC_API_KEY="nvapi-YOUR-KEY-HERE"
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

**Step 3: Pull the base image and build**
```bash
# Pull the base PyTorch image (~15 GB)
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# Build the local container (takes 15-30 minutes)
docker build --tag parakeet-tdt-v3-spark:latest -f docker/Dockerfile .
```

**Step 4: Start the stack**
```bash
# Save your API key for Docker Compose
echo "NGC_API_KEY=$NGC_API_KEY" > .env
chmod 600 .env

# Start the local build
docker compose -f docker/docker-compose.yml --env-file .env up -d
```

---

## Deploy via Portainer

If you prefer a Web UI instead of the terminal:

1. Open Portainer: `https://your-spark-ip:9443`
2. Go to **Stacks → Add Stack**
3. Name: `voice-agent`
4. **Web editor** — paste the contents of the `docker-compose.yml` file from Path A above.
5. **Environment variables** → Add:
   - `NGC_API_KEY` = `nvapi-your-actual-key`
6. Click **Deploy the stack**

---

## Testing & API Reference

### GET /health
Check if the model is loaded and ready.
```bash
curl -s http://localhost:8010/health | python3 -m json.tool
```

### POST /v1/audio/transcriptions
OpenAI Whisper-compatible endpoint.

```bash
# Transcribe an audio file (any format: wav, mp3, flac, ogg, m4a...)
curl -s http://localhost:8010/v1/audio/transcriptions \
    -F file="@your-audio.wav" \
    -F language=auto \
    | python3 -m json.tool
```

**Parameters (multipart form):**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Audio file |
| `language` | string | `auto` | Language code: `en`, `de`, `fr`, etc. |
| `response_format` | string | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |

---

## Integration Examples

### Open WebUI

In Open WebUI settings → Audio → STT:
- **Engine:** OpenAI
- **API Base URL:** `http://your-spark-ip:8010/v1`
- **Model:** `whisper-1`

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://your-spark-ip:8010/v1"
)

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="parakeet-tdt-0.6b-v3",
        file=f,
        language="de",
    )

print(result.text)
```

---

## Performance

Based on community benchmarks on DGX Spark:

| Metric | Value |
|--------|-------|
| Real-time Factor | ~15x (audio processed 15x faster than real-time) |
| GPU Memory | ~4.7 GB per inference |
| GPU Utilization | 90-96% during inference |
| Max segment (full attention) | 24 minutes |
| Max audio (local attention) | 3+ hours (auto-chunked) |
| Startup time | ~60-90 seconds |

---

## File Structure

```text
parakeet-spark/
├── docker/
│   ├── Dockerfile              # ARM64 build for DGX Spark
│   └── docker-compose.yml      # Local build compose file
├── requirements.txt            # Python dependencies
├── build-and-deploy.sh         # Automated setup script (for local builds)
├── .env.example                # Environment template
└── app/
    ├── main.py                 # FastAPI server + OpenAI endpoints
    └── transcriber.py          # NeMo model wrapper + chunking
```
