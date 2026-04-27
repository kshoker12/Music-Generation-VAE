# MusicGen Dashboard (web)

This UI is **static** (GitHub Pages) and calls your **RunPod FastAPI** endpoint for generation.

## First-time GitHub Pages

1. In the GitHub repo: **Settings → Pages → Build and deployment**.
2. Set **Source** to **GitHub Actions** (not “Deploy from a branch”).
3. Merge or push to **`main`** or **`master`** so [`.github/workflows/deploy_web.yml`](../.github/workflows/deploy_web.yml) runs (it only triggers on changes under `web/` or that workflow file, or on **Actions → Deploy web dashboard → Run workflow**).
4. After a successful run, the site is served at  
   `https://<owner>.github.io/<repo>/`  
   (project site). Vite uses `base: "./"` in [vite.config.ts](vite.config.ts) so assets load correctly under that path.

## API env vars (optional until the backend exists)

The dashboard **builds without** `VITE_*` set. The **Generate** button only needs them at runtime (they are baked in at build time by Vite).

- **UI-only deploy**: omit repository secrets; the build still succeeds; **Generate** will error until you add secrets and redeploy.
- **With API**: add repository **Settings → Secrets and variables → Actions** secrets `VITE_API_BASE_URL` and `VITE_API_KEY` (must match the inference service `API_KEY`, sent as `X-API-Key`). They are passed into the build job in the workflow.

When building locally:

- **`VITE_API_BASE_URL`**: e.g. `https://<your-runpod-endpoint>` or `http://127.0.0.1:8000` (no trailing slash required)
- **`VITE_API_KEY`**: must match the API service `API_KEY`

Local dev example:

```bash
export VITE_API_BASE_URL="http://127.0.0.1:8000"
export VITE_API_KEY="devkey"

npm --prefix web install --no-package-lock
npm --prefix web run dev -- --host 127.0.0.1 --port 5173
```

