from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.generate import router as generate_router


def _parse_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGIN", "http://127.0.0.1:5173")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or ["http://127.0.0.1:5173"]


app = FastAPI(title="musicgen-inference-api", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router)


@app.get("/health")
def health() -> dict:
    return {"ok": True}
