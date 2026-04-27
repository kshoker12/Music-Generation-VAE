from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.schemas import GenerateBody
from app.security import require_api_key

router = APIRouter()


def _ensure_ml_on_path() -> None:
    import os

    env_root = os.environ.get("REPO_ROOT")
    if env_root:
        repo_root = Path(env_root).resolve()
    else:
        here = Path(__file__).resolve()
        repo_root = None
        for p in [here.parents[i] for i in range(0, 10)]:
            if (p / "ml" / "src" / "musicgen").is_dir():
                repo_root = p
                break
        if repo_root is None:
            raise RuntimeError("Could not locate repo root containing ml/src/musicgen")

    ml_src = repo_root / "ml" / "src"
    s = str(ml_src)
    if s not in sys.path:
        sys.path.insert(0, s)


@router.post("/generate")
def generate(body: GenerateBody, _: None = Depends(require_api_key)) -> Response:
    _ensure_ml_on_path()

    from musicgen.inference.service import GenerateRequest, generate_midi_bytes

    req = GenerateRequest(
        model_type=body.model_type,
        attributes=body.attributes,
        seed=int(body.seed),
        temperature=float(body.temperature),
        top_p=float(body.top_p),
        block_size=int(body.block_size),
        bars_per_sample=int(body.bars_per_sample),
        ckpt_path=body.ckpt_path,
    )
    try:
        midi_bytes = generate_midi_bytes(req)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"{e}. Provide `ckpt_path` in the request body or set one of "
                "`DEFAULT_CKPT_PLAIN`, `DEFAULT_CKPT_VAE`, `DEFAULT_CKPT_SIMPLE_VAE` "
                "to a valid checkpoint path."
            ),
        ) from e

    # Persist a copy of every generation for later inspection/debugging.
    dumps_root = Path(os.environ.get("DUMPS_DIR", "dumps")).resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = dumps_root / f"{ts}_{body.model_type}_seed{body.seed}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "generated.mid").write_bytes(midi_bytes)
    (folder / "attributes.txt").write_text(json.dumps(body.attributes, indent=2), encoding="utf-8")

    filename = f"generated_{body.model_type}_seed{body.seed}.mid"
    return Response(
        content=midi_bytes,
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
