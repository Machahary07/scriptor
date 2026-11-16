import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.app import app
from engine.models import Style
from engine.models.font import FontModel
from engine.strokes import stroke3_to_polylines

client = TestClient(app)


def sample_files(n=6, slant=0.3):
    """Fake tablet samples: font-written lines with a strong slant, in tablet pixels."""
    rng = np.random.default_rng(0)
    texts = ["the cat sat on the mat", "a quick test of my pen", "notes from the lab today",
             "seven students wrote", "graphs and circuits", "measure it twice"]
    files = []
    for i in range(n):
        polys = stroke3_to_polylines(FontModel().write_line(texts[i % len(texts)], Style(slant=slant), 1.0, rng))
        strokes = [[[float(x) * 40 + 100, float(y) * 40 + 200, j] for j, (x, y) in enumerate(p)] for p in polys]
        files.append(("files", (f"s{i}.json", json.dumps({"text": texts[i % len(texts)], "strokes": strokes}),
                                "application/json")))
    return files


def test_writer_needs_consent():
    r = client.post("/api/writers", data={"name": "No Consent", "consent": "false"})
    assert r.status_code == 403


def test_full_flow_with_font_model():
    r = client.post("/api/writers", data={"name": "Test Writer", "consent": "true"}, files=sample_files())
    assert r.status_code == 200, r.text
    w = r.json()
    assert w["samples"] == 6

    r = client.post(f"/api/writers/{w['id']}/style", json={"mode": "measure"})
    assert r.status_code == 200, r.text
    assert r.json()["style"]["slant"] > 0.15  # picked up the slanted samples

    r = client.post("/api/generate", json={"text": "Hello from Scriptor.", "writer_id": w["id"], "seed": 1})
    assert r.status_code == 200, r.text
    g = r.json()
    paths = g["pages"][0]["paths"]
    assert len(paths) > 5

    r = client.post("/api/gcode", json={"paths": paths, "job": g["job"]})
    assert r.status_code == 200
    body = r.json()
    assert body["gcode"].startswith(f"; scriptor job={g['job']}")
    assert body["stats"]["paths"] > 0

    j = client.get(f"/api/jobs/{g['job']}").json()
    assert j["status"] == "done" and "page1.gcode" in j["files"]

    r = client.post("/api/generate", json={"text": "Test Writer", "writer_id": w["id"]})
    assert r.status_code == 422  # looks like a signature

    assert client.delete(f"/api/writers/{w['id']}").status_code == 200
    assert client.get(f"/api/writers/{w['id']}").status_code == 404


def test_unavailable_model_reports_409():
    r = client.post("/api/generate", json={"text": "hello", "model": "diffink"})
    assert r.status_code == 409


@pytest.mark.slow
def test_graves_with_lora_end_to_end():
    torch = pytest.importorskip("torch")  # noqa: F841
    from engine.config import WRITERS_DIR
    from training.personalise_lora import main as personalise
    from training.pretrain import main as pretrain

    pretrain(["--synthetic", "60", "--epochs", "1", "--hidden", "32", "--batch", "16"])
    w = client.post("/api/writers", data={"name": "Lora Writer", "consent": "true"}, files=sample_files()).json()
    client.post(f"/api/writers/{w['id']}/style", json={"mode": "measure"})
    personalise(["--writer", w["id"], "--epochs", "1", "--rank", "2"])
    assert (WRITERS_DIR / w["id"] / "adapter.pt").exists()

    r = client.post("/api/generate", json={"text": "hi there", "writer_id": w["id"], "model": "graves"})
    assert r.status_code == 200, r.text
    assert len(r.json()["pages"][0]["paths"]) > 0
