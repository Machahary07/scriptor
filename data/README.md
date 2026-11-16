# data/

Runtime data, created by the engine and never committed.

- `scriptor.db` — SQLite: writers (with consent record) and jobs
- `writers/<id>/samples/` — a writer's tablet lines (`.json`) and photos
- `writers/<id>/style.json`, `samples_norm.npz` — measured style and normalised samples
- `writers/<id>/adapter.pt` — LoRA adapter, after personalisation
- `models/graves_base.pt` — pre-trained LSTM-MDN weights (from `training/pretrain.py`)
- `jobs/<id>/` — G-code of generated pages and training logs

Set `SCRIPTOR_DATA` to keep this somewhere else.
