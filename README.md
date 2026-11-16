# Scriptor

AI handwriting synthesis with a virtual pen-plotter simulator.

Scriptor writes any text as real pen strokes in a chosen writer's style, turns the
strokes into G-code for a CoreXY pen plotter, and runs that G-code on a 3D virtual
plotter in the browser. The same G-code file can later drive a physical GRBL machine
unchanged.

```
text + writer samples ──► engine (Python, FastAPI) ──► G-code ──► virtual plotter (Next.js, three.js)
                          plan · generate · simplify             parse · CoreXY steps · time
                          join · order · write                   ink model · 3D build / plot
```

B.Tech CSE minor project, RSET, Assam Royal Global University.

## Quick start

Requirements: Python 3.10+, Node.js 20+.

```bash
# 1. Engine
python -m venv .venv
.venv/Scripts/activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # for CPU-only PyTorch add: --extra-index-url https://download.pytorch.org/whl/cpu
uvicorn engine.app:app --port 8000

# 2. Web app (second terminal)
cd web
npm install
npm run dev                       # http://localhost:3000
```

The web app forwards `/api/*` to the engine (`ENGINE_URL`, default `http://127.0.0.1:8000`).

Without the web app, from the command line:

```bash
python -m engine.cli examples/page.txt -o out/demo        # page1.gcode, page1.svg, stats.json
```

## What is in the box

| Part | Where | Status |
|---|---|---|
| Page planner: wrapping, margins, `[box 40 caption]` diagram boxes, multi-page | `engine/planner.py` | working |
| Font generator: Hershey script font with per-letter variation, neatness bias, writer slant and width | `engine/models/font.py` | working, default model |
| LSTM-MDN baseline with attention window and style encoder (Graves) | `engine/models/graves_net.py` | code complete and tested; needs training |
| LoRA adapters for per-writer personalisation | `engine/models/lora.py`, `training/personalise_lora.py` | working |
| SDT and DiffInk | `engine/models/external.py` | hooks for external commands (`SCRIPTOR_SDT_CMD`, `SCRIPTOR_DIFFINK_CMD`) |
| Photo to strokes (InkSight) | `engine/convert.py` | hook (`SCRIPTOR_INKSIGHT_CMD`) |
| Toolpath: RDP simplification, stroke joining, nearest-neighbour ordering | `engine/paths.py` | working |
| GRBL G-code writer with job trace tag | `engine/gcode.py` | working |
| Writers, consent, samples, style measurement, jobs (SQLite) | `engine/store.py` | working |
| Signature refusal | `engine/pipeline.py` | working |
| Studio, Writers (pen/mouse capture, uploads) and Simulator pages | `web/src/app/` | working |
| G-code parser, CoreXY kinematics, timing, validation | `web/src/simulator/` | working |
| Ink model and 3D plotter (build, exploded, plot, top views) | `web/src/simulator/` | working |

## Training the neural model

Training needs a GPU; free Google Colab is enough.

1. Register at the [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)
   and download `lineStrokes`, `ascii` and `original`.
2. Pre-train (about 30 epochs with early stopping):
   ```bash
   python training/pretrain.py --data /path/to/IAM-OnDB
   ```
   This writes `data/models/graves_base.pt`. The engine then lists the LSTM-MDN model as ready.
3. Personalise a writer once they have 10 to 20 transcribed lines (recorded in the Writers page):
   ```bash
   python training/personalise_lora.py --writer <writer-id>
   ```
   or press **Train LoRA adapter** in the Writers page.
4. Evaluate:
   ```bash
   python training/evaluate.py --model graves --writer <id> --pages 50 --cer
   ```

To check the training loop on a laptop without IAM-OnDB:
`python training/pretrain.py --synthetic 400 --epochs 3 --hidden 64`.

## Tests

```bash
pytest                          # engine: toolpath, G-code, planner, API, LoRA end to end
cd web && npm test              # simulator: parser, CoreXY steps, timing, validation, ink model,
                                # and agreement with the engine's statistics
```

## Machine assumptions

CoreXY, 200-step motors, 1/16 microstepping, 20-tooth GT2 pulleys (80 steps/mm), draw
feed 3000 mm/min, travel 5000 mm/min, pen down 0.15 s, pen up 0.10 s, A4 sheet with
the origin at its bottom-left corner. Pen lift uses `M3 S90` / `M5` on the spindle PWM
pin, which needs a servo-enabled GRBL build.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/models` | Models and whether each is ready |
| GET, POST | `/api/writers` | List or register writers (consent required; files optional) |
| GET, DELETE | `/api/writers/{id}` | Writer details, or delete the writer and all their data |
| POST | `/api/writers/{id}/samples` | Upload tablet `.json` / IAM `.xml` lines or photos |
| POST | `/api/writers/{id}/style` | `{"mode": "measure"}` or `{"mode": "lora"}` |
| POST | `/api/generate` | Text, writer, model, bias, seed → strokes for each page |
| POST | `/api/gcode` | Strokes → simplified, joined, ordered G-code and statistics |
| GET | `/api/jobs/{id}` | Job status, statistics, files, training log |

## Ethics

Writers must consent before any sample is stored, and only a writer's own samples build
their style. Signature-like requests are refused, and every G-code file carries a
`; scriptor job=…` trace line. Deleting a writer removes all their data. Machine-written
pages should not be submitted as handwritten work.
