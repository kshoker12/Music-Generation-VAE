from __future__ import annotations

import base64
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import runpod


def _ensure_ml_on_path() -> None:
    """
    Ensure `import musicgen...` works inside the container.
    """
    repo_root = os.environ.get("REPO_ROOT", "/app")
    ml_src = Path(repo_root) / "ml" / "src"
    s = str(ml_src)
    if s not in sys.path:
        sys.path.insert(0, s)


_ensure_ml_on_path()

# Warm-start: import + tokenizer init are cached across requests
print("RunPod cold start: importing musicgen inference service...")
try:
    from musicgen.inference.service import GenerateRequest, generate_midi_bytes  # noqa: E402

    _IMPORT_OK = True
    print("Cold start OK: musicgen imports succeeded.")
except Exception as e:
    _IMPORT_OK = False
    _IMPORT_ERR = str(e)
    print(f"Cold start FAILED: {e}")
    traceback.print_exc()


def _err(message: str, *, detail: Any | None = None) -> Dict[str, Any]:
    return {"error": message, "detail": detail}


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler.

    Expected request shape (similar spirit to your chess project):

    {
      "input": {
        "endpoint": "generate",
        "model_type": "vae" | "simple_vae" | "plain",
        "attributes": [[...],[...],...],   // [8,4] ints 0..7
        "seed": 123,
        "temperature": 1.0,
        "top_p": 0.9,
        "block_size": 1024,
        "bars_per_sample": 8,
        "ckpt_path": "ml/src/musicgen/runs/vae_v3/ckpt.pt"  // optional
      }
    }

    Returns:
      Success: {"midi_b64": "...", "mime_type": "audio/midi"} — RunPod wraps this under "output"
        in the HTTP response (clients read output.midi_b64).

      Error: {"error": "...", "detail": ...} — appears as output.error / output.detail.
    """
    if not _IMPORT_OK:
        return _err("Model code failed to import on cold start.", detail=_IMPORT_ERR)

    inp = (event or {}).get("input") or {}
    endpoint = inp.get("endpoint", "generate")
    if endpoint != "generate":
        return _err("Unknown endpoint.", detail={"endpoint": endpoint, "supported": ["generate"]})

    try:
        req = GenerateRequest(
            model_type=str(inp.get("model_type", "vae")),  # type: ignore[arg-type]
            attributes=inp["attributes"],
            seed=int(inp.get("seed", 0)),
            temperature=float(inp.get("temperature", 1.0)),
            top_p=float(inp.get("top_p", 0.9)),
            block_size=int(inp.get("block_size", 1024)),
            bars_per_sample=int(inp.get("bars_per_sample", 8)),
            ckpt_path=inp.get("ckpt_path"),
        )
    except Exception as e:
        return _err("Invalid request input.", detail=str(e))

    try:
        midi_bytes = generate_midi_bytes(req)
        midi_b64 = base64.b64encode(midi_bytes).decode("ascii")
        return {"midi_b64": midi_b64, "mime_type": "audio/midi"}
    except Exception as e:
        traceback.print_exc()
        return _err("Generation failed.", detail=str(e))


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

