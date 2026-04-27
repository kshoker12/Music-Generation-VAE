# RunPod Serverless (handler) — MusicGen

This repo supports **RunPod Serverless** using a Python **handler** (`handler.py`) that RunPod calls once per request.

## What this service does

- Cold start: imports your `musicgen` code and keeps a model cache warm across requests (inside the same container lifecycle).
- Request: generates a MIDI file (bytes) and returns it as base64.

## Build + push (Linux amd64)

From repo root:

```bash
docker build --platform linux/amd64 -f services/runpod_serverless/Dockerfile -t <dockerhub_user>/musicgen-runpod:<tag> .
docker push <dockerhub_user>/musicgen-runpod:<tag>
```

## RunPod endpoint setup

In RunPod UI:

- Create a **Serverless Endpoint**
- Use your pushed container image
- Make sure the container command is the default (`python -u handler.py`) or set it explicitly

## Request shape

This handler dispatches on `event["input"]["endpoint"]` (currently only `"generate"`).

Example request JSON:

```json
{
  "input": {
    "endpoint": "generate",
    "model_type": "vae",
    "attributes": [[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3],[3,3,3,3]],
    "seed": 123,
    "temperature": 1.0,
    "top_p": 0.9,
    "block_size": 1024,
    "bars_per_sample": 8,
    "ckpt_path": "ml/src/musicgen/runs/vae_v3/ckpt.pt"
  }
}
```

## Response shape

The handler returns a **flat** dict; RunPod's API wraps it under `output`.

**Handler return (success):**

```json
{
  "midi_b64": "<base64-encoded-midi-bytes>",
  "mime_type": "audio/midi"
}
```

**Typical RunPod `runsync` / job JSON** (outer fields vary by API version):

```json
{
  "status": "COMPLETED",
  "output": {
    "midi_b64": "<base64-encoded-midi-bytes>",
    "mime_type": "audio/midi"
  }
}
```

**Handler return (error):** `{ "error": "<message>", "detail": ... }` — surfaced under `output.error` / `output.detail` in the job response.

## Notes

- **Checkpoints**: this image bakes in `ml/` from the repo. If you want to mount checkpoints elsewhere, pass `ckpt_path` as an absolute path or set `REPO_ROOT` appropriately.
- **Torch/CUDA**: `services/runpod_serverless/requirements.txt` uses the PyTorch CUDA 12.1 wheel index. If you change the CUDA base image, update the extra-index-url accordingly.

