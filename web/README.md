# MusicGen Dashboard (web)

This UI is **static** (GitHub Pages) and calls your **RunPod FastAPI** endpoint for generation.

## Required build-time env vars (Vite)

Set these when building `web/`:

- **`VITE_API_BASE_URL`**: e.g. `https://<your-runpod-endpoint>` (no trailing slash required)
- **`VITE_API_KEY`**: must match the API service `API_KEY` (sent as `X-API-Key`)

Local dev example:

```bash
export VITE_API_BASE_URL="http://127.0.0.1:8000"
export VITE_API_KEY="devkey"

npm --prefix web install --no-package-lock
npm --prefix web run dev -- --host 127.0.0.1 --port 5173
```

GitHub Pages:
- Vite uses `base: "./"` for subpath hosting.
- Inject `VITE_API_BASE_URL` + `VITE_API_KEY` via GitHub Actions secrets at build time.

