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

## Notes → model

Learns your handwriting from phone photos of handwritten pages, on free Colab (T4), in one notebook:
[`notebooks/notes_to_model.ipynb`](notebooks/notes_to_model.ipynb).

1. Photograph 10 or more pages: from straight above, good light, no shadow, one page per photo.
2. Either put the photos in a Google Drive folder named `scriptor_photos`, or keep them on your laptop and pick them when cell 2 asks.
3. Open [Colab](https://colab.research.google.com) → File → Open notebook → GitHub → `Machahary07/scriptor`,
   branch `notes-to-model` (or `main` after merge) → `notebooks/notes_to_model.ipynb`.
4. Runtime → Change runtime type → **T4 GPU**.
5. Run the cells in order. Allow Google Drive access when asked.
6. If cell 3 (InkSight setup) fails, set `INK_PINS = "card"` in that cell and run it again. It stops after 20 minutes and prints the exact error; it never fakes strokes.
7. Look at the previews printed by cell 4 (photo vs recovered strokes). Every word and the reason it was kept or flagged is in `photo2ink/words.json`.
8. Outputs land in `MyDrive/Scriptor/notes_run/`: `loss_curve.png`, `generated_bias0.png`, `generated_bias2.png`,
   `generated_bias*.gcode` (load them in the Simulator page), `run_summary.json`, `graves_base.pt`, `adapter.pt`.

What happens:

| Step | Code | Notes |
| --- | --- | --- |
| Clean pages | `engine/photo2ink/page.py` | crop to paper, deskew (projection profile), contrast, adaptive binarisation |
| Find and read words | docTR (free, local) | word boxes, text and confidence |
| Word image → strokes | `engine/photo2ink/inksight.py` | Google InkSight Small-p, in its own TensorFlow environment (`setup_env.sh`) |
| Check and group | `engine/photo2ink/layout.py` | a word is kept when its strokes cover its ink (F1 ≥ 0.3) and its text is trusted (docTR confident, or docTR and InkSight agree); words → lines; noisy lines dropped |
| Dataset | `training/notes_dataset.py` | repo preprocessing → stroke-3; 90/10 split by line; 3 jittered copies (scale, slant, rotation) of each training line |
| Stage A | `training/notes_train.py` | LSTM-MDN from scratch, 15 min, Adam 1e-3, clip 10 |
| Stage B | same | base frozen, LoRA r=8 on LSTM gates, attention window and output head, 10 min, Adam 5e-4, original lines only |
| Output | `engine/planner.py`, `engine/pipeline.py` | lines at bias 0 and 2 → PNG and G-code |

Without IAM-OnDB the base model only sees your own notes, so expect rough output after 25 minutes.
For readable text, pre-train stage A on IAM-OnDB (above) and use the notes only for the adapter.

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
