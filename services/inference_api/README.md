# Inference API (FastAPI) for RunPod Serverless

This service exposes a single endpoint:

- `POST /generate` → returns `audio/midi` bytes
- `GET /health` → `{ "ok": true }`

## Required environment variables

- **`API_KEY`**: shared secret; clients must send header `X-API-Key: <value>`
- **`ALLOWED_ORIGIN`**: comma-separated list of allowed browser origins for CORS
  - Example: `https://<user>.github.io`

## Checkpoint paths

Defaults (repo-relative inside the container):

- `DEFAULT_CKPT_PLAIN` (default `ml/src/musicgen/runs/plain_v1/ckpt.pt`)
- `DEFAULT_CKPT_VAE` (default `ml/src/musicgen/runs/vae_v1/ckpt.pt`)
- `DEFAULT_CKPT_SIMPLE_VAE` (default `ml/src/musicgen/runs/simple_vae_v1/ckpt.pt`)

Override these on RunPod if you mount checkpoints elsewhere.

## Repo root discovery

The API adds `ml/src` to `PYTHONPATH`. If auto-discovery fails, set:

- **`REPO_ROOT`**: absolute path to the repo root containing `ml/src/musicgen`

The Docker image sets `REPO_ROOT=/app`.

## Local run (dev)

From repo root:

```bash
export API_KEY="devkey"
export ALLOWED_ORIGIN="http://127.0.0.1:5173"
cd services/inference_api
../../.venv/bin/uvicorn app.main:app --reload --port 8000
```

## Docker build + run

From repo root:

```bash
docker build -f services/inference_api/Dockerfile -t musicgen-inference-api .
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e API_KEY="devkey" \
  -e ALLOWED_ORIGIN="http://127.0.0.1:5173" \
  musicgen-inference-api
```

## Example curl

```bash
curl -L -o out.mid \
  -H "X-API-Key: devkey" \
  -H "Content-Type: application/json" \
  -d '{"model_type":"vae","seed":123,"temperature":1.0,"top_p":0.9,"attributes":[[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3]]}' \
  http://127.0.0.1:8000/generate
```

## RunPod Serverless notes

RunPod’s exact “HTTP handler” wiring varies by template. This image runs **uvicorn** and listens on **`PORT`** (defaults to `8000`).

On RunPod, set:

- `PORT` (if your platform injects it)
- `API_KEY`
- `ALLOWED_ORIGIN` to your GitHub Pages origin

For GPU, prefer a CUDA-capable base image and install the matching `torch` wheel in `requirements.txt` (the default `python:3.12-slim` image is CPU-oriented).
