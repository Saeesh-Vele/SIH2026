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
       -> vqa_grounding_node | change_node | fusion_node
       -> output_combiner -> trace_logger
```

| Node | What it does |
| --- | --- |
| `intake` | Resolves the upload id to files on disk |
| `intent_classifier` | Labels the question with an LLM over OpenRouter |
| `input_validator` | Checks the bound imagery can answer that intent |
| `task_router` | Maps intent to task and pulls the model off the registry |
| `vqa_grounding_node` | Runs zero-shot LLaVA-1.5 for VQA and captioning |
| `change_node` | Diffs a bi-temporal pair and describes what moved |
| `fusion_node` | Encodes an optical+SAR pair, fuses it, and answers from both |
| `specialist_stub` | Unreachable today — the landing place for a task added to the enum before its node exists |
| `output_combiner` | Settles `{answer, evidence, confidence, model_used}` |
| `trace_logger` | Writes the run to `execution_traces` |

Two conditional edges short-circuit inference: a rejected input skips to
`output_combiner`, and `task_router` picks the specialist. `output_combiner` and
`trace_logger` always run, so every request — answered, rejected or failed —
leaves a complete trace.

A run reports one of four statuses, and only `ok` carries an answer:
`rejected` (the imagery does not match the question), `unavailable` (the model
could not be loaded here), `failed` (a node raised).

**Dispatch reads the query and the imagery.** The classifier labels the
question, and then `reconcile_with_upload` checks that label against what is
actually bound. Two of the three upload modes admit exactly one intent — a
bi-temporal pair can only be compared over time, an optical+SAR pair can only be
fused — so a label those rule out is corrected rather than left to be rejected,
and the trace records the swap. Single-image mode admits two intents, so it
never triggers: a change question asked of one image is a real mismatch, and the
validator still says so.

**Intents.** The classifier picks one of `single_image_vqa`,
`single_image_captioning`, `change_vqa` or `optical_sar_fusion`. Both
single-image intents run on the same checkpoint — captioning differs in the
prompt, not the model: `single_image_vqa` answers a pointed question, and
`single_image_captioning` widens the scope to the whole scene via
`CAPTION_SUFFIX` in `app/agent/nodes/vqa.py`. The split is scope, not subject.
Set `SATQUERY_OPENROUTER_API_KEY` to use an LLM;
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
| `vqa_grounding` | Answers questions about one scene, and captions it |
| `change_detection` | Compares two captures of one footprint and masks what changed |
| `optical_sar_fusion` | Combines an optical scene with SAR when cloud blocks the optical |

All three return the same object — `{answer, evidence, confidence, model_used}` —
so the combiner, the API and the canvas never branch on which one ran. What
varies is the evidence: boxes for VQA, masks for change detection, none for
fusion, which produces a described scene rather than geometry.

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

**The VQA checkpoint.** The zero-shot base is `llava-hf/llava-1.5-7b-hf`,
loaded through transformers' generic LLaVA classes (`_load_llava` in
`app/models/geochat.py`). One checkpoint serves both single-image intents:
`single_image_vqa` asks it a pointed question, `single_image_captioning` asks it
to describe the whole scene. LLaVA-1.5 captions natively, which is why
captioning — not grounding — is the second mandatory single-image task.

The engine keeps a second loader path, `--loader geochat`, for GeoChat-format
checkpoints — a fine-tuned adapter, say — and picks it automatically when the
`geochat` package is importable. Nothing in the code is GeoChat-specific beyond
that path and the grounding-token parser.

*Why not GeoChat itself.* GeoChat is a LLaVA-1.5 derivative and would be the
better remote-sensing base, but `MBZUAI/geochat-7B` is not usable here today:
its official repo pins a 2023-era dependency stack (torch 2.0.1, transformers
4.31.0) that conflicts with everything else this project runs on, and its
published weights do not load correctly through transformers' generic LLaVA
classes — the checkpoint layout does not match that architecture. Stock
LLaVA-1.5 loads cleanly on the modern stack, so it is the working base until a
GeoChat-format checkpoint arrives in a form the current environment can load.

*On grounding.* The problem statement asks for captioning **or** grounding
alongside VQA, and this project answers with captioning. Region grounding would
need a grounding-trained checkpoint of its own — LLaVA-1.5 was never trained to
emit the `{<x1><y1><x2><y2>}` box tokens `parse_grounding` reads, so it returns
prose and an empty `evidence` list. Adding it later is a checkpoint swap, not a
rewrite: point `vqa_grounding.base_model` at something like
`RogerFerrod/GroundSet-LLaVA-1.6-7B`, restore a `single_image_grounding` intent,
and the parser, the `evidence` field and the canvas overlay layer are already
wired to draw the boxes. That is a possible addition once the core requirements
are complete, not part of them.

The task key stays named `vqa_grounding` so the YAML, the `TaskType` enum and
the frontend's `TaskType` union keep matching; only the intent taxonomy changed.

### Change detection

A prompted diff, and one checkpoint does both halves:

- **Where.** Both frames go through the VQA model's own vision tower. Patch
  embeddings are compared cosine-wise, giving a difference map on the encoder's
  patch grid (24x24 for LLaVA-1.5's CLIP ViT-L/14-336). Cells above
  `diff_threshold` are grouped into connected regions, and each region becomes
  one `mask` overlay — its bounding rectangle in 0-1 scene space, with the mean
  difference score as its confidence. A rectangle rather than a traced contour
  on purpose: the grid is coarse, and a pixel-accurate outline would imply
  precision the features do not carry. Regions below `min_region_area` are
  dropped as speckle.
- **What.** Both frames then go to the VQA model with a change-focused
  instruction, and a change-VQA question is appended after it so "did the pier
  get longer?" is answered rather than replaced by a generic summary.

No second checkpoint, so changing `base_model` changes the detector and the
describer together. The honest caveat: LLaVA-1.5 was trained on single images.
It accepts two and answers, but its comparison is weaker than a model trained on
pairs. The difference map does not depend on that — the vision tower is applied
to each frame separately.

### Optical-SAR fusion

Encode, fuse, verbalise, answer:

1. **Two** SSL4EO-S12 encoders, one per modality. `encoder` is the timm
   **architecture** and is shared (`vit_base_patch16_224`);
   `optical_checkpoint_path` and `sar_checkpoint_path` are the **pretraining
   runs** that fill it. They are not interchangeable — each patch embedding is
   built for a band count, so the Sentinel-1 encoder cannot read Sentinel-2 and
   vice versa. Both are ViT-B/16 and pool to 768, so concat gives 1536;
   `FusionEngine.load` checks that rather than assuming it.
2. The embeddings are fused — `fusion_strategy: concat` by default, since it
   loses nothing and needs no learned head. `gated` weights each modality by its
   share of the magnitude; `cross_attention` needs a trained head and is
   rejected rather than silently approximated.
3. The fused vector is reduced to a few interpretable numbers — which sensor
   dominates, whether the two agree — and rendered as one sentence using the
   phrases in `feature_vocabulary`. This is the join between a vector nobody can
   read and a model that only reads text, and it is deliberately a small
   inspectable mapping rather than a learned captioner: **every phrase the VQA
   model is told comes from the YAML**, so a claim the imagery cannot support
   cannot appear without someone writing it there.
4. That sentence goes to the VQA model alongside the optical image and the
   question. The sentence is also surfaced in the result panel — it is half the
   reasoning, so hiding it would make the answer harder to check.

The encoder is the one part that needs real weights. Until `checkpoint_path`
points at some, the node reports itself unavailable: a ViT-Base/16 with random
weights would still produce an embedding, and that embedding would still yield a
confident-sounding sentence. Dropping real weights in is a path in
`model_config.yaml` and nothing else.

```bash
mkdir -p backend/weights/fusion
curl -L -o backend/weights/fusion/ssl4eo_s12_vitb16.pth \
  https://huggingface.co/wangyi111/SSL4EO-S12/resolve/main/B2_vitb16_mae_ep99.pth

python backend/tests/verify_fusion_checkpoint.py
```

The verifier builds the architecture twice, loads each checkpoint into its own
instance, and separates the leftovers into three kinds — missing, unexpected and
shape mismatches — because they mean different things. Both are clean:

| | bands | tensors | missing | unexpected | mismatch | matched |
| --- | --- | --- | --- | --- | --- | --- |
| optical (`251k_ms.lmdb`) | 13 | 259 | 0 | 109 | 0 | 150 |
| SAR (`251k_sar.lmdb`) | 2 | 254 | 0 | 104 | 0 | 150 |

The unexpected keys are the MAE decoder and mask token, which exist only during
pretraining; the optical checkpoint adds three `hog.*` buffers from the MAE-MFP
training variant. Both pool to 768d, so concat gives 1536d.

Weights are gitignored; the URLs above are the source of truth.

**Band counts are read, not assumed.** Sentinel-1 is 2 bands (VV, VH) and
Sentinel-2 is 13 (L1C) or 12 (L2A); none of them is RGB's 3. Built at timm's
default the tensor shapes collide and the patch embedding stays randomly
initialised, which is why a shape mismatch is a failure rather than a warning and
why `load` checks each checkpoint's own `patch_embed.proj.weight` before building
anything. `FusionConfig` also rejects the old single `checkpoint_path` outright
rather than letting one modality be encoded by the other's weights.

**Still needed before real inference**, both of which are input-side rather than
model-side — the encoders themselves load and run:

- *A multispectral reader.* PIL has no mode for 13-band or 2-band rasters, so
  `_read_bands` tries rasterio, then tifffile, then PIL, and refuses with a
  named reason rather than handing back a silently truncated three-band version
  of a thirteen-band scene. `pip install rasterio` closes it.
- *Per-band normalisation statistics.* `_to_tensor` falls back to 0.5/0.5 for any
  band count it has no statistics for, which centres the input without
  pretending to know the sensor. SSL4EO-S12 publishes per-band values, and they
  belong in `model_config.yaml` before these embeddings are trusted.

**Running it locally.** LLaVA-1.5-7B in 4-bit needs a CUDA GPU. To exercise the
whole path without one, run unquantized on CPU:

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

## Tests

```bash
python backend/tests/test_cpu_smoke.py      # graph paths, upload validation, routing
python backend/tests/test_specialists.py    # change detection, fusion, dispatch
```

CPU only, and dependency-free beyond what the app already needs — no GPU, no
MongoDB, no OpenRouter key, no network, no weights. Engines are exercised
through their pure parts (the difference map, the fusion arithmetic, the
verbalisation) and the nodes against recording stubs registered on the model
registry, which is the same seam real checkpoints arrive through. Both files
also run under `pytest backend/tests` if you have it installed.

## Zero-shot inference test

Confirms the VQA checkpoint runs in 4-bit before any of it is wired into the
backend. Needs a CUDA GPU: bitsandbytes has no 4-bit kernel for CPU or MPS.

```bash
pip install -r scripts/requirements.txt
python scripts/test_geochat_zeroshot.py \
  --image path/to/scene.png \
  --question "How many aircraft are visible on the tarmac?"
```

It defaults to `llava-hf/llava-1.5-7b-hf` and takes any LLaVA-compatible
checkpoint via `--model`. `--loader` picks the code path: `llava` for
transformers' generic LLaVA classes, `geochat` for the official `geochat`
package, `auto` (the default) for `geochat` when it is importable and `llava`
otherwise.

The script and the controller share one implementation, `app/models/geochat.py`,
so what this verifies is what the graph runs. Both return the same shape:

```json
{"answer": "...", "evidence": [...], "confidence": 0.87, "model_used": "..."}
```

`confidence` is the mean probability of the tokens the model chose — a
generative VQA model has no calibrated confidence head, so this is the honest
stand-in. `evidence` holds grounding boxes parsed out of
`{<x1><y1><x2><y2>|<angle>}` output tokens and normalised to 0–1 of the scene
extent, which is what the canvas draws — empty under stock LLaVA-1.5, which
does not emit them, and populated for a GeoChat-format checkpoint.
