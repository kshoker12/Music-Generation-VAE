# MusicGen Dashboard (web)

Static **GitHub Pages** UI that calls **RunPod Serverless** (`runsync`) for generation (no FastAPI in this path).

## First-time GitHub Pages

1. In the GitHub repo: **Settings → Pages → Build and deployment**.
2. Set **Source** to **GitHub Actions**.
3. Push changes under `web/` or run **Actions → Deploy web dashboard → Run workflow**.
4. Site URL: `https://<owner>.github.io/<repo>/`. [vite.config.ts](vite.config.ts) uses `base: "./"` for subpath assets.

## RunPod build-time env (required for Generate)

Vite bakes these in at `npm run build`. Add them as **repository Actions secrets** (Settings → Secrets and variables → Actions), then redeploy the web workflow so a new build runs:

| Secret | Example | Meaning |
|--------|---------|--------|
| `VITE_RUNPOD_ENDPOINT_ID` | `y1vs0tn7rp44mp` | RunPod serverless **endpoint ID** (path segment in `https://api.runpod.ai/v2/<id>/runsync`) |
| `VITE_RUNPOD_API_KEY` | (RunPod API key) | Sent as `Authorization: Bearer …` |

The workflow [`.github/workflows/deploy_web.yml`](../.github/workflows/deploy_web.yml) passes these into the build step as `env`.

**Security:** anyone can read `VITE_*` values from the published JavaScript bundle. Do not use a production RunPod key you cannot rotate or cap.

**CORS:** the browser calls `https://api.runpod.ai` from `https://<user>.github.io`. If the console shows a CORS error, RunPod’s API must allow your origin, or you need a same-origin proxy (outside this repo’s static-only path).

## Local dev

```bash
export VITE_RUNPOD_ENDPOINT_ID="y1vs0tn7rp44mp"
export VITE_RUNPOD_API_KEY="your_runpod_key"

npm --prefix web install --no-package-lock
npm --prefix web run dev -- --host 127.0.0.1 --port 5173
```

The **Generate** button calls `POST https://api.runpod.ai/v2/<id>/runsync` with body `{ "input": { "endpoint": "generate", ... } }` and decodes `output.midi_b64` (with fallback for legacy nested `output.output.midi_b64` until workers pick up a newer handler image).

## RunPod container (handler)

Build and deploy the GPU image from repo root: see [services/runpod_serverless/README.md](../services/runpod_serverless/README.md). After changing `handler.py`, rebuild and push the image so RunPod workers return the flat `output.midi_b64` shape.
