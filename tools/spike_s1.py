"""Spike S1 — Tripo behaviour validation (plan 18 §S1, WP-14a).

Textures a small UV-mapped mesh through the real Tripo API in each guidance mode and records what
matters for the pipeline: whether vertex positions and UVs survive (route decision), the output GLB
structure, credits per call and task durations. Spends credits: roughly 10-20 per mode.

Usage (needs ``TRIPO_API_KEY`` in the environment or ``.env``)::

    uv run --env-file .env python tools/spike_s1.py --out docs/spikes/s1 \
        --fbx tests/fixtures/fbx/rigged_cube.fbx --modes text text+style image images
    uv run python tools/spike_s1.py --dry-run    # prepare + request building only, no API calls

Outputs ``<out>/findings.json`` and ``<out>/S1-tripo-findings.md``; downloaded GLBs stay in
``<out>/<mode>/`` for inspection in Blender.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import SecretStr
from ulid import ULID

from grounded_pipeline.blender.protocol import JobRequest, PrepareModelData, PrepareModelParams
from grounded_pipeline.blender.runner import SubprocessBlenderRunner, default_entry_script
from grounded_pipeline.domain.errors import PipelineError
from grounded_pipeline.domain.library_config import (
    TRIPO_DEFAULT_BASE_URL,
    TRIPO_DEFAULT_MODEL_VERSION,
    GeometrySection,
    TripoSection,
    UvSection,
    VerifySection,
)
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.gltf.loader import load_glb
from grounded_pipeline.gltf.verify import verify_surfaces
from grounded_pipeline.pipeline.timing import SystemClock, SystemSleeper
from grounded_pipeline.texturing.provider import GuidanceMode, TextureRequest
from grounded_pipeline.texturing.tripo_provider import TripoTextureProvider
from grounded_pipeline.tripo.client import TripoClient
from grounded_pipeline.tripo.polling import RateLimiter

DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
PROMPT = (
    "Stylized hand-painted game prop, smooth soft shading, saturated vibrant colours, "
    "clean readable shapes, no outlines, PBR. A small wooden crate with brass corners."
)


@dataclass
class ModeResult:
    mode: str
    ok: bool
    task_id: str | None = None
    credits_consumed: float | None = None
    duration_s: float | None = None
    status: str | None = None
    error: str | None = None
    route: str | None = None
    position_match: bool | None = None
    uv_match: bool | None = None
    topology_match: bool | None = None
    normalize_scale: float | None = None
    normalize_translation: list[float] | None = None
    reason: str | None = None
    glb: dict[str, Any] = field(default_factory=dict)


def _write_png(path: Path, size: int = 512) -> Path:
    """A deterministic gradient reference image (Pillow is available in the venv)."""
    from PIL import Image  # noqa: PLC0415 - tool-only dependency use

    img = Image.new("RGB", (size, size))
    px = img.load()
    assert px is not None
    for y in range(size):
        for x in range(size):
            px[x, y] = (int(255 * x / size), 120, int(255 * y / size))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def prepare_static_glb(fbx: Path, out_dir: Path, blender: Path, levels: int) -> PrepareModelData:
    """Run the real ``prepare_model`` job to obtain ``static.glb`` and the prepare report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = SubprocessBlenderRunner(
        executable=blender,
        entry_script=default_entry_script(),
        semaphore=threading.BoundedSemaphore(1),
        env=dict(os.environ),
    )
    params = PrepareModelParams(
        source_fbx=fbx.resolve(),
        out_blend=out_dir / "prepared.blend",
        out_static_glb=out_dir / "static.glb",
        out_report=out_dir / "prepare_report.json",
        geometry=GeometrySection(subdivision_levels=levels, shapekeys="drop"),
        uv=UvSection(mode="auto"),
        material_name="s1_mat",
    )
    result = runner.run(
        JobRequest(
            job="prepare_model",
            job_id=str(ULID()),
            model_id=None,
            params=params.model_dump(mode="json"),
        ),
        timeout_s=600,
        log_path=out_dir / "prepare.log",
        cancel=CancelToken(),
    )
    return PrepareModelData.model_validate(result.data)


def describe_glb(path: Path) -> dict[str, Any]:
    """Structure summary of a GLB for the findings document."""
    doc = load_glb(path)
    return {
        "bytes": path.stat().st_size,
        "primitives": len(doc.meshes),
        "triangles": int(sum(len(m.indices) for m in doc.meshes)),
        "has_uv": all(m.uvs is not None for m in doc.meshes),
        "materials": [
            {
                "name": m.name,
                "base_color_texture": m.base_color_texture is not None,
                "metallic_roughness_texture": m.metallic_roughness_texture is not None,
                "normal_texture": m.normal_texture is not None,
                "occlusion_texture": m.occlusion_texture is not None,
                "emissive_factor": list(m.emissive_factor),
                "metallic_factor": m.metallic_factor,
                "roughness_factor": m.roughness_factor,
            }
            for m in doc.materials
        ],
        "images": [{"mime": i.mime_type, "bytes": len(i.data), "name": i.name} for i in doc.images],
    }


def build_request(
    mode: GuidanceMode, static_glb: Path, style_png: Path, quality: str, seed: int
) -> TextureRequest:
    guidance: dict[str, Path] = {}
    if mode is GuidanceMode.IMAGE:
        guidance = {"front": style_png}
    elif mode is GuidanceMode.IMAGES:
        guidance = {"front": style_png, "left": style_png}
    text_modes = {GuidanceMode.TEXT, GuidanceMode.TEXT_STYLE}
    sha = hashlib.sha256(f"s1:{mode.value}:{seed}".encode()).hexdigest()
    return TextureRequest(
        model_id="s1_cube",
        mesh_path=static_glb,
        mode=mode,
        prompt=PROMPT if mode in text_modes else None,
        style_image=style_png if mode is GuidanceMode.TEXT_STYLE else None,
        guidance_images=guidance,
        model_version=TRIPO_DEFAULT_MODEL_VERSION,
        quality=quality,
        pbr=True,
        bake=True,
        delight=True,
        alignment="original_image",
        seed=seed,
        request_sha256=sha,
    )


def _constant_dir(path: Path) -> Callable[[TextureRequest], Path]:
    def pick(_req: TextureRequest) -> Path:
        return path

    return pick


def run_mode(
    provider: TripoTextureProvider,
    req: TextureRequest,
    prep: PrepareModelData,
    static_glb: Path,
) -> ModeResult:
    res = ModeResult(mode=req.mode.value, ok=False)
    started = time.monotonic()
    try:
        tr = provider.texture(req, resume_task_id=None, cancel=CancelToken())
    except PipelineError as exc:
        res.error = f"{exc.code}: {exc.message} {exc.detail}"
        res.duration_s = round(time.monotonic() - started, 1)
        return res
    res.duration_s = round(time.monotonic() - started, 1)
    res.task_id = tr.task_id
    res.credits_consumed = tr.credits_consumed
    res.status = str(tr.raw_output.get("status"))
    output = tr.raw_output.get("output") or {}
    res.normalize_scale = output.get("normalize_scale")
    res.normalize_translation = output.get("normalize_translation")
    try:
        report = verify_surfaces(
            static_glb,
            tr.result_glb,
            VerifySection(),
            expected_triangles=prep.static_glb_triangles,
            expected_unique_positions=prep.static_glb_unique_positions,
            normalize_scale=res.normalize_scale,
            normalize_translation=tuple(res.normalize_translation)
            if res.normalize_translation
            else None,
        )
        res.route = report.route
        res.position_match = report.position_match
        res.uv_match = report.uv_match
        res.topology_match = report.topology_match
        res.reason = report.reason
    except PipelineError as exc:
        res.reason = f"{exc.code}: {exc.message}"
    res.glb = describe_glb(tr.result_glb)
    res.ok = True
    return res


def render_markdown(results: list[ModeResult], prep: PrepareModelData, meta: dict[str, Any]) -> str:
    lines = [
        "# S1 — Tripo behaviour validation (findings)",
        "",
        f"Date: {meta['date']} · mesh: `{meta['fbx']}` subdivided {meta['levels']}x -> "
        f"{prep.static_glb_triangles} tris, {prep.static_glb_unique_positions} unique positions, "
        f"UV decision `{prep.uv.decision}` · quality `{meta['quality']}` · "
        f"model `{meta['model_version']}`",
        "",
        "| mode | ok | status | credits | seconds | route | pos | uv | topo "
        "| normalize_scale | images |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        imgs = ",".join(sorted({i["mime"] for i in r.glb.get("images", [])})) or "-"
        lines.append(
            f"| {r.mode} | {r.ok} | {r.status or r.error} | {r.credits_consumed} | "
            f"{r.duration_s} | "
            f"{r.route} | {r.position_match} | {r.uv_match} | {r.topology_match} | "
            f"{r.normalize_scale} | {imgs} |"
        )
    lines += ["", "## Per-mode detail", ""]
    for r in results:
        lines += [f"### {r.mode}", "", "```json", json.dumps(asdict(r), indent=2), "```", ""]
    lines += [
        "## Decisions to record in plan 18",
        "",
        "- R-01 (UV/topology preservation): see `uv`/`pos` columns → extract vs bake default.",
        "- R-02 (image/images accepted for file_token input): see `ok`/`status` of `image`, "
        "`images`.",
        "- R-03 (GLB structure): see `images` mime types and material texture presence.",
        "- Credits: compare `credits` with `[tripo.credit_table]` defaults (10/10/20/30).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fbx", type=Path, default=Path("tests/fixtures/fbx/rigged_cube.fbx"))
    parser.add_argument("--out", type=Path, default=Path("docs/spikes/s1"))
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--quality", default="standard")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--modes", nargs="+", default=["text", "text+style", "image", "images"])
    parser.add_argument("--dry-run", action="store_true", help="no API calls")
    args = parser.parse_args(argv)

    out: Path = args.out
    prep = prepare_static_glb(args.fbx, out / "prepare", args.blender, args.levels)
    static_glb = out / "prepare" / "static.glb"
    style_png = _write_png(out / "style.png")
    sys.stderr.write(
        f"prepared: {prep.static_glb_triangles} tris, "
        f"{prep.static_glb_unique_positions} positions, "
        f"uv={prep.uv.decision}\n"
    )
    requests = [
        build_request(GuidanceMode(m), static_glb, style_png, args.quality, args.seed)
        for m in args.modes
    ]
    if args.dry_run:
        for req in requests:
            sys.stderr.write(f"dry-run {req.mode.value}: {req.model_dump(mode='json')}\n")
        return 0

    key = os.environ.get("TRIPO_API_KEY")
    if not key:
        sys.stderr.write("TRIPO_API_KEY not set\n")
        return 2
    clock, sleeper = SystemClock(), SystemSleeper()
    tripo_cfg = TripoSection()
    client = TripoClient(
        SecretStr(key),
        TRIPO_DEFAULT_BASE_URL,
        sleeper=sleeper,
        clock=clock,
        rate_limiter=RateLimiter(
            tripo_cfg.max_concurrent_tasks,
            tripo_cfg.min_request_interval_s,
            clock=clock,
            sleeper=sleeper,
        ),
        record_dir=out / "recorded",
    )
    before = client.balance()
    results: list[ModeResult] = []
    for req in requests:
        mode_dir = out / req.mode.value.replace("+", "_")
        mode_dir.mkdir(parents=True, exist_ok=True)
        provider = TripoTextureProvider(
            client,
            credit_table=tripo_cfg.credit_table,
            poll_interval_s=tripo_cfg.poll_interval_s,
            poll_timeout_s=tripo_cfg.poll_timeout_s,
            presign_above_mb=tripo_cfg.use_presign_above_mb,
            work_dir_for=_constant_dir(mode_dir),
        )
        sys.stderr.write(f"-> {req.mode.value} ...\n")
        res = run_mode(provider, req, prep, static_glb)
        sys.stderr.write(
            f"   {res.status or res.error} credits={res.credits_consumed} route={res.route}\n"
        )
        results.append(res)
    after = client.balance()
    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "fbx": str(args.fbx),
        "levels": args.levels,
        "quality": args.quality,
        "model_version": TRIPO_DEFAULT_MODEL_VERSION,
        "balance_before": before.balance,
        "balance_after": after.balance,
    }
    (out / "findings.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "prepare": prep.model_dump(mode="json"),
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "S1-tripo-findings.md").write_text(
        render_markdown(results, prep, meta), encoding="utf-8"
    )
    sys.stderr.write(f"spent {before.balance - after.balance} credits; findings in {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
