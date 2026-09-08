# SatQuery AI

An agentic vision–language assistant for remote-sensing image analysis. Ask a
question about satellite imagery in plain language; a LangGraph controller
classifies the question, checks it against the imagery you bound, runs the model
that fits, and streams every step it took into an execution trace.

```
SIH2026/
├── frontend/   Next.js console — image canvas, query input, trace feed
├── backend/    FastAPI service — /upload, /query, /health, MongoDB, ModelRegistry
└── scripts/    Standalone experiments, run outside the service
```

The frontend talks to the backend for real — upload, query and trace all go over
HTTP. Everything below the controller degrades rather than breaks: no MongoDB, no
LLM key and no GPU each cost one capability and are reported in the trace, not
raised as errors.

## The controller

```
intake -> intent_classifier -> input_validator -> task_router
       -> vqa_grounding_node | specialist_stub
       -> output_combiner -> trace_logger
```

| Node | What it does |
| --- | --- |
| `intake` | Resolves the upload id to files on disk |
| `intent_classifier` | Labels the question with an LLM over OpenRouter |
| `input_validator` | Checks the bound imagery can answer that intent |
| `task_router` | Maps intent to task and pulls the model off the registry |
| `vqa_grounding_node` | Runs zero-shot GeoChat — the one real specialist today |
| `specialist_stub` | Stands in for change detection and fusion until Phase 4 |
| `output_combiner` | Settles `{answer, evidence, confidence, model_used}` |
| `trace_logger` | Writes the run to `execution_traces` |

Two conditional edges short-circuit inference: a rejected input skips to
`output_combiner`, and `task_router` picks the specialist. `output_combiner` and
`trace_logger` always run, so every request — answered, rejected or failed —
leaves a complete trace.

A run reports one of four statuses, and only `ok` carries an answer:
`rejected` (the imagery does not match the question), `unavailable` (the model
could not be loaded here), `failed` (a node raised).

**Intents.** The classifier picks one of `single_image_vqa`,
`single_image_grounding`, `change_vqa` or `optical_sar_fusion`. Both
single-image intents run on the same checkpoint — grounding differs in the
prompt, not the model. Set `SATQUERY_OPENROUTER_API_KEY` to use an LLM;
`SATQUERY_OPENROUTER_MODEL` selects DeepSeek or Gemini. Without a key the
controller falls back to keyword classification and says so in the trace.

**Adding a specialist.** Register a loader and the graph picks it up:

```python
registry.register_loader("change_detection", load_my_model)
```

## Tasks

Task names are shared across all three layers — the YAML keys, the Pydantic
`TaskType` enum, and the frontend's `TaskType` union all use the same strings.

| Task | What it does |
| --- | --- |
| `vqa_grounding` | Answers questions about one scene and localises what it names |
| `change_detection` | Compares two captures of one footprint |
| `optical_sar_fusion` | Combines an optical scene with SAR when cloud blocks the optical |

## Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Set `NEXT_PUBLIC_API_BASE_URL` if the backend is not on `localhost:8000`; see
`.env.example`. Queries go to `/query/stream`, so trace steps appear as the
controller produces them rather than all at once at the end.

Georeferenced fields in the scene readout show a dash: `/upload` stores the file
but does not read GeoTIFF headers yet, and a plausible-looking coordinate nobody
measured would be worse than an empty one.

Upload accepts three modes — single image, optical + SAR pair, bi-temporal pair.
GeoTIFF is the working format; PNG and JPEG pass only when **benchmark dataset
mode** is on, matching the same rule in `backend/app/routes/upload.py`.

## Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload    # http://localhost:8000/docs
```

MongoDB is optional at boot — the service starts without it and `/health`
reports `mongo: false`. Point it elsewhere with `SATQUERY_MONGO_URI`; see
`.env.example` for the full set of `SATQUERY_`-prefixed settings.

**Collections**

- `query_history` — query text, timestamp, task_type, asset ids
- `execution_traces` — task_selected, models_used, parameters, confidence, steps, timestamp
- `uploads` — staged assets per upload id

**Endpoints**

| Route | Purpose |
| --- | --- |
| `POST /api/upload` | Stores the imagery, returns an `upload_id` |
| `POST /api/query` | Runs the controller, returns the result |
| `POST /api/query/stream` | Same, as SSE: `plan`, then `step` per node, then `result` |
| `GET /api/query/history` | Recent queries |
| `GET /api/query/{id}/trace` | The stored execution trace |
| `GET /api/health` | Version, Mongo reachability, configured tasks |

**Model registry.** `model_config.yaml` declares one entry per task.
`vqa_grounding` resolves to a real `GeoChatEngine`; tasks with no registered
loader resolve to a `MockModel`. Constructing the engine is cheap — no weights
are touched until the first question — so the service starts anywhere and
reports a missing GPU only when someone actually asks something.

**Running a small model locally.** GeoChat-7B in 4-bit needs a CUDA GPU. To
exercise the whole path without one, point the registry at any LLaVA-family
checkpoint:

```yaml
# model_config.local.yaml
vqa_grounding:
  base_model: llava-hf/llava-1.5-7b-hf
  quantization: none
  device: cpu
```

```bash
SATQUERY_MODEL_CONFIG_PATH=./model_config.local.yaml uvicorn app.main:app --reload
```

## Zero-shot inference test

Confirms GeoChat runs in 4-bit before any of it is wired into the backend.
Needs a CUDA GPU: bitsandbytes has no 4-bit kernel for CPU or MPS.

```bash
pip install -r scripts/requirements.txt
python scripts/test_geochat_zeroshot.py \
  --image path/to/scene.png \
  --question "How many aircraft are visible on the tarmac?"
```

It prefers the official `geochat` package when it is importable and otherwise
falls back to transformers' generic LLaVA classes, which load the same weights
for plain VQA. Force either path with `--loader geochat|llava`.

The script and the controller share one implementation, `app/models/geochat.py`,
so what this verifies is what the graph runs. Both return the same shape:

```json
{"answer": "...", "evidence": [...], "confidence": 0.87, "model_used": "..."}
```

`confidence` is the mean probability of the tokens the model chose — a
generative VQA model has no calibrated confidence head, so this is the honest
stand-in. `evidence` holds GeoChat's grounding boxes, parsed out of its
`{<x1><y1><x2><y2>|<angle>}` output tokens and normalised to 0–1 of the scene
extent, which is what the canvas draws.
