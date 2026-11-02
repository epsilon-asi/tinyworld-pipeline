# 17 — Future Generative Pipeline (architecture and contracts only)

**Status:** not in v1 scope. This document fixes the shape so v1 code needs no rework when it arrives.

## 1. Goal

From our own style sources (style images + base prompts) produce brand-new game-ready creatures:

```
style sources ──▶ concept images ──▶ item extraction ──▶ multi-view sheets ──▶ mesh generation
             ──▶ texture (reuse v1) ──▶ rig / animation ──▶ export + Unity handoff (reuse v1)
```

Human review gates sit after concept generation and after item extraction (choosing what becomes an
asset is a creative decision, not an algorithmic one).

## 2. Folder layout (`library/generative/`, reserved in 05 §7)

```
generative/
├── concepts/<batch_id>/
│   ├── batch.md                 # front matter: provider, model, seeds, count, style refs; body: ## prompt(s)
│   ├── images/<n>.png           # generated concept images
│   ├── items.json               # extraction results: bounding boxes / masks per detected item
│   └── review.json              # human decisions: approved image+item pairs → item_ids
└── items/<item_id>/
    ├── item.md                  # front matter: source batch/image/bbox, display_name, overrides; body: ## prompt, ## notes
    ├── concept.png              # the cropped/cleaned item image
    ├── multiview/{front,left,back,right}.png
    ├── mesh/model.glb           # Tripo multiview-to-model output (untextured or textured)
    ├── rig/rigged.glb           # Tripo rig output (+ retargeted clips)
    ├── work/  output/  .pipeline/   # same conventions as models/<id>/
```

An **item** becomes a **model** at the export boundary: `output/<item_id>/` has the identical layout and
`manifest.json` schema as `models/<id>/output/`, so the Unity package needs no changes.

## 3. Stage graph (DAG; `pipeline/graph.py` must support fan-out/fan-in)

```
concept_generate (batch) ──▶ item_extract (batch) ──▶ [REVIEW GATE] ──▶ multiview (item)
   ──▶ mesh_generate (item) ──▶ [optional decimate/retopo] ──▶ texture (item; v1 stage reused)
   ──▶ verify/apply (reused) ──▶ rig (item) ──▶ retarget (item) ──▶ export_generated (item) ──▶ handoff (reused)
```

Batch-level and item-level stages have different `ModelContext` equivalents (`BatchContext`,
`ItemContext`); `Stage.run` signature becomes generic over the context type. In v1, `Stage` is typed to
`ModelContext` only; the migration is a `Generic[TContext]` parameter — recorded as a known change.

## 4. Provider ports (new Protocols; same style as `TextureProvider`)

```python
class ImageProvider(Protocol):
    def text_to_image(self, req: TextToImageRequest) -> ImageResult
    def image_to_image(self, req: ImageToImageRequest) -> ImageResult        # style transfer onto a sketch
class SegmentationProvider(Protocol):                                        # local model or API
    def extract_items(self, image: Path, hints: ExtractHints) -> list[ItemCandidate]  # bbox, mask, label, score
class MultiviewProvider(Protocol):
    def image_to_multiview(self, req: MultiviewRequest) -> MultiviewResult       # 4 views
    def edit_multiview(self, req: EditMultiviewRequest) -> MultiviewResult
class MeshProvider(Protocol):
    def multiview_to_model(self, req: MeshRequest) -> MeshResult                  # task id + glb
    def decimate(self, req: DecimateRequest) -> MeshResult
class RigProvider(Protocol):
    def rig_check(self, mesh_task_or_token: str) -> RigCheckResult                # rig_type
    def rig(self, req: RigRequest) -> RigResult                                   # spec "mixamo" | …
    def retarget(self, req: RetargetRequest) -> RetargetResult                    # preset clips
```

Tripo implements all of these today via the endpoints in appendix A §1 (`/generation/text-to-image`,
`/generation/image-to-image`, `/generation/image-to-multiview`, `/generation/edit-multiview`,
`/generation/multiview-to-model`, `/mesh/decimate`, `/animations/rig-check`, `/animations/rig`,
`/animations/retarget`). The `TripoClient` from WP-05 gains one method per endpoint; the envelope,
polling, upload, budget and rate-limit code is shared unchanged.

## 5. Data contracts to add (sketch; finalised when scheduled)

- `BatchFrontMatter { schema_version, name, provider, image_model, count, seeds[], style_refs[], aspect }`
- `ItemCandidate { image, bbox[x,y,w,h], mask_path?, label, score }`
- `ReviewDecision { image, candidate_index, item_id, approved_by, at }`
- `ItemFrontMatter { schema_version, display_name, origin{batch_id,image,bbox}, mesh{face_limit,
  quad, model_version}, rig{spec, clips[]}, texture overrides (same as models), unity overrides }`
- `manifest.provenance` gains `origin: {kind: "generated", batch_id, item_id, mesh_task_id, rig_task_id}`
  (additive; `schema_version` stays 1 if all new fields have defaults; else bump to 2 with reader fallback).

## 6. Review gate mechanics

A gate is a stage whose `run` returns `StageOutcome(status="skipped", extra={"awaiting":"review"})` until
`review.json` contains decisions for every candidate; `gp review <batch>` opens a small local web page
(static HTML + JSON, no server) or accepts a CSV. Fingerprint of downstream item stages includes the
review decision, so re-approving re-runs exactly what changed.

## 7. Reuse map (what v1 already provides)

| Future need | v1 component |
|---|---|
| Upload/poll/download/budget/rate-limit | `tripo/*` |
| Texturing a generated mesh | `texture`, `verify`, `apply` stages (input is `mesh/model.glb` instead of `static.glb`) |
| Unity delivery | `export`-style FBX conversion (Blender job `glb_to_fbx_rigged`, new), `handoff`, C# package |
| State, caching, concurrency, CLI patterns | `pipeline/*`, `cli/*` |
| Style prompts and style images | `style/` + `prompt_builder` (template gains `{concept_prompt}`) |

## 8. Known changes required when this is scheduled

1. `Stage` generic over context type; `StageGraph` DAG scheduling with per-node concurrency.
2. `library/discovery` learns `generative/` trees; `gp run --kind models|items|batches`.
3. Blender job `glb_to_fbx_rigged` (import rigged GLB → FBX with `FBX_EXPORT_KWARGS`) + RFC relaxed
   (no source rig to compare against; compare rig vs Tripo's rig fingerprint instead).
4. New error codes `E_GEN_*`, documented in 14.
