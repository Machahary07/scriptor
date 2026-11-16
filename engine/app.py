"""FastAPI service for the engine. Run: uvicorn engine.app:app --port 8000"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import store
from .config import JOBS_DIR, ROOT
from .convert import ConversionUnavailable, inksight_available, photo_to_samples
from .models import ModelUnavailable, get_model, list_models
from .pipeline import Refused, check_request, generate_strokes, make_gcode
from .planner import PageSpec

app = FastAPI(title="Scriptor engine", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/models")
def models():
    return list_models()


@app.get("/api/consent")
def consent():
    return {"text": store.CONSENT_TEXT}


# ---- writers -----------------------------------------------------------------

@app.get("/api/writers")
def writers():
    return store.list_writers()


@app.post("/api/writers")
async def create_writer(name: str = Form(...), consent: bool = Form(False), files: list[UploadFile] = File(default=[])):
    try:
        w = store.create_writer(name.strip(), consent)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    for f in files:
        await _store_upload(w["id"], f)
    return store.get_writer(w["id"])


@app.get("/api/writers/{wid}")
def get_writer(wid: str):
    return _writer_or_404(wid)


@app.delete("/api/writers/{wid}")
def delete_writer(wid: str):
    _writer_or_404(wid)
    store.delete_writer(wid)
    return {"deleted": wid}


@app.post("/api/writers/{wid}/samples")
async def add_samples(wid: str, files: list[UploadFile] = File(...)):
    _writer_or_404(wid)
    added = 0
    for f in files:
        added += await _store_upload(wid, f)
    return {"added": added, "writer": store.get_writer(wid)}


class StyleRequest(BaseModel):
    mode: str = Field("measure", pattern="^(measure|lora)$")


@app.post("/api/writers/{wid}/style")
def build_style(wid: str, req: StyleRequest = StyleRequest()):
    _writer_or_404(wid)
    _convert_photos(wid)
    try:
        style = store.build_style(wid)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if req.mode == "measure":
        return {"style": style}
    ok, why = get_model("graves").available()
    if not ok:
        raise HTTPException(409, f"LoRA needs the base model: {why}")
    jid = store.create_job(wid, "graves", {"kind": "lora"})
    log = JOBS_DIR / jid / "train.log"

    def run():
        cmd = [sys.executable, str(ROOT / "training" / "personalise_lora.py"), "--writer", wid]
        with open(log, "w") as fh:
            res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
        store.finish_job(jid, error=None if res.returncode == 0 else f"exit code {res.returncode}")

    threading.Thread(target=run, daemon=True).start()
    return {"style": style, "job": jid}


# ---- generation ----------------------------------------------------------------

class GenerateRequest(BaseModel):
    text: str = Field(..., max_length=20000)
    writer_id: str | None = None
    model: str = "font"
    bias: float = Field(1.0, ge=0.0, le=10.0)
    seed: int = 0
    x_height: float = Field(2.6, ge=1.0, le=8.0)
    line_spacing: float = Field(8.0, ge=4.0, le=20.0)


@app.post("/api/generate")
def generate(req: GenerateRequest):
    writer = _writer_or_404(req.writer_id) if req.writer_id else None
    try:
        check_request(req.text, writer["name"] if writer else None)
        style = store.load_style(req.writer_id)
    except Refused as e:
        raise HTTPException(422, f"refused: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    jid = store.create_job(req.writer_id, req.model, req.model_dump(exclude={"text"}))
    spec = PageSpec(x_height=req.x_height, line_spacing=req.line_spacing)
    try:
        pages = generate_strokes(req.text, style, req.model, req.bias, req.seed, spec)
    except (ModelUnavailable, KeyError) as e:
        store.finish_job(jid, error=str(e))
        raise HTTPException(409, str(e))
    out = [{"layout": p.layout, "paths": [q.round(3).tolist() for q in p.paths]} for p in pages]
    store.finish_job(jid, {"pages": len(out)})
    return {"job": jid, "pages": out}


class GcodeRequest(BaseModel):
    paths: list[list[list[float]]]
    page_height: float = 297.0
    order: bool = True
    simplify_mm: float = Field(0.1, ge=0.0, le=2.0)
    join_mm: float = Field(0.3, ge=0.0, le=2.0)
    job: str | None = None
    page: int = 1


@app.post("/api/gcode")
def gcode(req: GcodeRequest):
    tag = f"job={req.job}" if req.job else "job=adhoc"
    res = make_gcode(req.paths, req.page_height, tag, req.order, req.simplify_mm, req.join_mm)
    if req.job and (JOBS_DIR / req.job).exists():
        (JOBS_DIR / req.job / f"page{req.page}.gcode").write_text(res.gcode)
    return {"gcode": res.gcode, "stats": res.stats}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = store.get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    log = JOBS_DIR / jid / "train.log"
    if log.exists():
        j["log_tail"] = log.read_text(errors="replace")[-2000:]
    return j


@app.get("/api/jobs/{jid}/files/{name}")
def job_file(jid: str, name: str):
    p = (JOBS_DIR / jid / name).resolve()
    if p.parent != (JOBS_DIR / jid).resolve() or not p.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(p)


# ---- helpers -----------------------------------------------------------------

def _writer_or_404(wid: str) -> dict:
    w = store.get_writer(wid)
    if not w:
        raise HTTPException(404, "writer not found")
    return w


async def _store_upload(wid: str, f: UploadFile) -> int:
    data = await f.read()
    try:
        return store.add_samples(wid, data, f.filename or "sample.json")
    except ValueError as e:
        raise HTTPException(400, str(e))


def _convert_photos(wid: str) -> None:
    """Convert stored photos to stroke samples when InkSight is configured."""
    if not inksight_available():
        return
    sd = store.writer_dir(wid) / "samples"
    for img in [p for p in sd.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]:
        done = img.with_suffix(img.suffix + ".converted")
        if done.exists():
            continue
        try:
            for i, s in enumerate(photo_to_samples(img)):
                (sd / f"{img.stem}_{i}.json").write_text(json.dumps(s))
            done.write_text("ok")
        except ConversionUnavailable:
            pass
