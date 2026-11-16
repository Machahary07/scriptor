"""Paths and settings shared by the engine and the training scripts."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SCRIPTOR_DATA", ROOT / "data"))
WRITERS_DIR = DATA_DIR / "writers"
MODELS_DIR = DATA_DIR / "models"
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "scriptor.db"
GRAVES_WEIGHTS = MODELS_DIR / "graves_base.pt"

for d in (WRITERS_DIR, MODELS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)
