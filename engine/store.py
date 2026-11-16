"""Writers, their samples and styles, and generation jobs.

Metadata lives in SQLite; samples, styles, adapters and job files live under
data/writers/<id>/ and data/jobs/<id>/. Nothing is stored without a consent
record, and delete_writer() removes everything a writer gave.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import DB_PATH, JOBS_DIR, WRITERS_DIR
from .models import Style
from .models.font import FontModel
from .strokes import estimate_slant, normalise_line, polylines_to_stroke3, resample

CONSENT_TEXT = (
    "I agree that Scriptor may store my handwriting samples on this machine to build a model of my "
    "handwriting style, that only I may generate text in my style, that signatures will not be generated, "
    "and that I can ask for my samples and style to be deleted at any time."
)

SCHEMA = """
create table if not exists writers (
  id text primary key, name text not null, consent_text text not null, consent_at real not null,
  created_at real not null);
create table if not exists jobs (
  id text primary key, writer_id text, model text, status text, created_at real, finished_at real,
  params text, stats text, error text);
"""


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def writer_dir(writer_id: str) -> Path:
    return WRITERS_DIR / writer_id


# ---- writers -----------------------------------------------------------------

def create_writer(name: str, consent: bool) -> dict:
    if not consent:
        raise PermissionError("consent is required before any sample is stored")
    wid = uuid.uuid4().hex[:10]
    now = time.time()
    with db() as con:
        con.execute("insert into writers values (?,?,?,?,?)", (wid, name, CONSENT_TEXT, now, now))
    (writer_dir(wid) / "samples").mkdir(parents=True, exist_ok=True)
    return get_writer(wid)


def get_writer(wid: str) -> dict | None:
    with db() as con:
        row = con.execute("select * from writers where id=?", (wid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    sd = writer_dir(wid) / "samples"
    d["samples"] = len(list(sd.glob("*.json"))) if sd.exists() else 0
    d["photos"] = len([p for p in sd.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]) if sd.exists() else 0
    style = writer_dir(wid) / "style.json"
    d["style"] = json.loads(style.read_text()) if style.exists() else None
    d["adapter"] = (writer_dir(wid) / "adapter.pt").exists()
    return d


def list_writers() -> list[dict]:
    with db() as con:
        ids = [r["id"] for r in con.execute("select id from writers order by created_at")]
    return [get_writer(i) for i in ids]


def delete_writer(wid: str) -> None:
    with db() as con:
        con.execute("delete from writers where id=?", (wid,))
    shutil.rmtree(writer_dir(wid), ignore_errors=True)


# ---- samples -----------------------------------------------------------------

def parse_sample(data: bytes, filename: str) -> list[dict]:
    """Read tablet samples. Returns [{"text", "strokes": [[[x, y, t], ...], ...]}].

    Accepted: Scriptor JSON (one object or a list of objects) and IAM-OnDB
    lineStrokes XML (text unknown, given as "").
    """
    name = filename.lower()
    if name.endswith(".json"):
        obj = json.loads(data)
        items = obj if isinstance(obj, list) else [obj]
        for it in items:
            if "strokes" not in it:
                raise ValueError("sample JSON needs a 'strokes' field")
        return [{"text": it.get("text", ""), "strokes": it["strokes"]} for it in items]
    if name.endswith(".xml"):
        root = ET.fromstring(data)
        strokes = [[[float(p.get("x")), float(p.get("y")), float(p.get("time", 0))] for p in s.iter("Point")]
                   for s in root.iter("Stroke")]
        return [{"text": "", "strokes": strokes}]
    raise ValueError(f"unsupported sample file: {filename}")


def add_samples(wid: str, data: bytes, filename: str) -> int:
    sd = writer_dir(wid) / "samples"
    sd.mkdir(parents=True, exist_ok=True)
    if Path(filename).suffix.lower() in (".jpg", ".jpeg", ".png"):
        # Photos are kept for offline-to-online conversion (InkSight); see engine/convert.py.
        (sd / f"{uuid.uuid4().hex[:8]}{Path(filename).suffix.lower()}").write_bytes(data)
        return 1
    items = parse_sample(data, filename)
    for it in items:
        if sum(len(s) for s in it["strokes"]) < 5:
            continue
        (sd / f"{uuid.uuid4().hex[:8]}.json").write_text(json.dumps(it))
    return len(items)


def load_samples(wid: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((writer_dir(wid) / "samples").glob("*.json"))]


def sample_polylines(sample: dict) -> list[np.ndarray]:
    return [np.asarray(s, dtype=np.float64)[:, :2] for s in sample["strokes"] if len(s) > 0]


# ---- style -------------------------------------------------------------------

def build_style(wid: str) -> dict:
    """Measure a writer's style from their samples and store it.

    The measured values (slant, letter width, spacing) drive the font generator.
    Normalised stroke-3 copies of the samples are saved for the neural models'
    style encoder and for LoRA personalisation.
    """
    samples = load_samples(wid)
    if not samples:
        raise ValueError("writer has no tablet samples yet")
    slants, widths, norm = [], [], []
    font = FontModel()
    for s in samples:
        polys = sample_polylines(s)
        if not polys:
            continue
        slants.append(estimate_slant(polys))
        npolys, _ = normalise_line(polys)
        npolys = [resample(p, 0.05) for p in npolys]
        norm.append(polylines_to_stroke3(npolys))
        if s.get("text"):
            w = max(p[:, 0].max() for p in npolys)
            ref = font.estimate_width(s["text"], Style(width=1.0, spacing=1.0, word_gap=1.0))
            if ref > 0:
                widths.append(w / ref)
    width = float(np.clip(np.median(widths), 0.7, 1.5)) if widths else 1.0
    style = {
        "slant": float(np.clip(np.median(slants), -0.4, 0.6)),
        "width": width,
        "spacing": 1.0,
        "word_gap": 1.0,
        "messiness": 1.0,
        "samples": len(norm),
        "built_at": time.time(),
    }
    wd = writer_dir(wid)
    np.savez_compressed(wd / "samples_norm.npz", *norm)
    (wd / "style.json").write_text(json.dumps(style, indent=2))
    return style


def load_style(wid: str | None) -> Style:
    if not wid:
        return Style()
    wd = writer_dir(wid)
    p = wd / "style.json"
    if not p.exists():
        raise ValueError("style has not been built for this writer")
    d = json.loads(p.read_text())
    st = Style(writer_id=wid, **{k: d[k] for k in ("slant", "width", "spacing", "word_gap", "messiness")})
    npz = wd / "samples_norm.npz"
    if npz.exists():
        with np.load(npz) as z:
            st.samples = [z[k] for k in z.files]
    if (wd / "adapter.pt").exists():
        st.adapter_path = str(wd / "adapter.pt")
    return st


# ---- jobs --------------------------------------------------------------------

def create_job(writer_id: str | None, model: str, params: dict) -> str:
    jid = uuid.uuid4().hex[:12]
    with db() as con:
        con.execute("insert into jobs (id, writer_id, model, status, created_at, params) values (?,?,?,?,?,?)",
                    (jid, writer_id, model, "running", time.time(), json.dumps(params)))
    (JOBS_DIR / jid).mkdir(parents=True, exist_ok=True)
    return jid


def finish_job(jid: str, stats: dict | None = None, error: str | None = None) -> None:
    with db() as con:
        con.execute("update jobs set status=?, finished_at=?, stats=?, error=? where id=?",
                    ("failed" if error else "done", time.time(), json.dumps(stats or {}), error, jid))


def get_job(jid: str) -> dict | None:
    with db() as con:
        row = con.execute("select * from jobs where id=?", (jid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["params"] = json.loads(d["params"] or "{}")
    d["stats"] = json.loads(d["stats"] or "{}")
    d["files"] = sorted(p.name for p in (JOBS_DIR / jid).glob("*")) if (JOBS_DIR / jid).exists() else []
    return d


def style_dict(style: Style) -> dict:
    d = asdict(style)
    d.pop("samples")
    return d
