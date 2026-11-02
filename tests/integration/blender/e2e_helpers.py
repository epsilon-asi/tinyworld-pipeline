"""Harness for the credit-free end-to-end suite (``test_end_to_end.py``).

Everything here is real production wiring except two swaps made on the assembled
:class:`PipelineContext`: the texture provider becomes a :class:`FakeTextureProvider` and the
``SubprocessBlenderRunner`` is wrapped in a job-name counting proxy. The GLB factories re-emit the
geometry the provider was sent (``work/static.glb``) with ``glb_builder``'s known-value PBR set.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from typer.testing import CliRunner, Result

from fakes.fake_texture_provider import FakeTextureProvider
from fakes.glb_builder import ImageSpec, MaterialSpec, from_mesh, textured_set
from grounded_pipeline.blender.locate import BlenderInfo
from grounded_pipeline.blender.protocol import JobRequest, JobResult
from grounded_pipeline.blender.runner import BlenderRunner
from grounded_pipeline.cli.app import build_app
from grounded_pipeline.cli.state import CliHooks, CliState
from grounded_pipeline.cli.wiring import (
    build_production_context,
    make_tripo_client,
    resolve_blender,
    secrets_from_mapping,
)
from grounded_pipeline.config.layering import CliOverrides
from grounded_pipeline.domain.library_config import BlenderSection
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.gltf.loader import load_glb
from grounded_pipeline.observability.console import Console, RecordingConsole
from grounded_pipeline.pipeline.context import PipelineContext
from grounded_pipeline.pipeline.runner import ContextFactory, RunSetup
from grounded_pipeline.texturing.provider import TextureRequest

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_LIBRARY = ROOT / "tests" / "fixtures" / "library_minimal"
MODEL = "cube_rigged"
CLIP = "walk"
CREDITS = 7.5
PROVIDER_SCALE = 2.0
FAST_BAKE_TOML = "\n[bake]\nresolution = 512\nsamples = 4\nmargin_px = 4\n"
URP_MAPS = ("basecolor", "normal", "metallic_smoothness", "occlusion", "emission")
FULL_RUN_JOBS = ["prepare_model", "export_rigged", "roundtrip_animation"]
BAKE_RUN_JOBS = ["prepare_model", "bake_textures", "export_rigged", "roundtrip_animation"]
BASECOLOR_RGB = (200, 100, 50)  # textured_set() base colour
COLOUR_TOLERANCE = 6


# --------------------------------------------------------------------------- GLB factories
def provider_material() -> tuple[list[ImageSpec], MaterialSpec]:
    """``textured_set()`` with a unit ``emissiveFactor`` (how a provider ships an emissive map).

    The builder's default factor is glTF's black default, which Blender's importer honours by
    wiring the texture with ``Emission Strength = 0``; the bake job then skips the emission map.
    """
    images, material = textured_set()
    return images, dataclasses.replace(material, emissive_factor=(1.0, 1.0, 1.0))


def input_geometry(req: TextureRequest) -> tuple[Any, Any, Any]:
    """``(positions, uvs, indices)`` of every primitive in the GLB the provider was sent."""
    doc = load_glb(req.mesh_path)
    positions, uvs, indices, offset = [], [], [], 0
    for mesh in doc.meshes:
        assert mesh.uvs is not None, "prepare must export UVs"
        positions.append(mesh.positions)
        uvs.append(mesh.uvs)
        indices.append(mesh.indices + offset)
        offset += len(mesh.positions)
    return np.concatenate(positions), np.concatenate(uvs), np.concatenate(indices)


def glb_textured_copy(req: TextureRequest) -> bytes:
    """The input surface untouched, with a full PBR texture set → ``extract`` route."""
    positions, uvs, indices = input_geometry(req)
    images, material = provider_material()
    return from_mesh(positions, uvs, indices, images=images, materials=[material])


def glb_scaled_reunwrapped(req: TextureRequest) -> bytes:
    """The input surface at ``PROVIDER_SCALE`` with flipped UVs → ``bake`` route."""
    positions, uvs, indices = input_geometry(req)
    images, material = provider_material()
    flipped = (1.0 - uvs).astype(np.float32)
    return from_mesh(
        positions, flipped, indices, scale=PROVIDER_SCALE, images=images, materials=[material]
    )


# --------------------------------------------------------------------------- harness
class CountingBlenderRunner:
    """Records every job name launched through the real runner (shared list across runs)."""

    def __init__(self, inner: BlenderRunner, jobs: list[str]) -> None:
        self._inner = inner
        self.jobs = jobs

    def run(
        self, req: JobRequest, *, timeout_s: int, log_path: Path, cancel: CancelToken
    ) -> JobResult:
        self.jobs.append(req.job)
        return self._inner.run(req, timeout_s=timeout_s, log_path=log_path, cancel=cancel)


def copy_library(dest: Path, *, extra_toml: str = "") -> Path:
    """A writable copy of ``library_minimal`` with optional ``library.toml`` additions."""
    shutil.copytree(FIXTURE_LIBRARY, dest)
    if extra_toml:
        toml = dest / "library.toml"
        toml.write_text(toml.read_text(encoding="utf-8") + extra_toml, encoding="utf-8")
    return dest


def hermetic_env() -> dict[str, str]:
    """What the CLI and the Blender child see: no ``.env``, no key, no ``GP_*`` overrides."""
    return {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR") if k in os.environ}


def production_factory(
    lib: Path,
    env: dict[str, str],
    blender: BlenderInfo,
    provider: FakeTextureProvider,
    jobs: list[str],
) -> ContextFactory:
    """The real production context with only ``provider`` (and a counting proxy) swapped in."""
    hooks = CliHooks(
        ctx_factory=None,
        tripo_client_factory=make_tripo_client,
        blender_resolver=resolve_blender,
        secrets_loader=lambda: secrets_from_mapping(env),
    )

    def _factory(setup: RunSetup) -> PipelineContext:
        state = CliState(
            library=lib,
            json=True,
            log_level=None,
            blender=blender.path,
            no_color=True,
            env=env,
            cwd=lib,
            console=Console.recording(),
            hooks=hooks,
        )
        ctx = build_production_context(
            setup, state=state, secrets=secrets_from_mapping(env), blender=blender
        )
        assert ctx.cli_overrides == CliOverrides().model_dump(mode="json")
        return dataclasses.replace(
            ctx, provider=provider, blender=CountingBlenderRunner(ctx.blender, jobs)
        )

    return _factory


@dataclass
class E2E:
    """One tmp library plus the ``gp`` app wired to it."""

    lib: Path
    provider: FakeTextureProvider
    blender: BlenderInfo
    env: dict[str, str]
    console: RecordingConsole
    app: Any
    jobs: list[str] = field(default_factory=list)
    runner: CliRunner = field(default_factory=CliRunner)
    first: dict[str, Any] = field(default_factory=dict)
    first_jobs: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, lib: Path, blender_exe: Path, provider: FakeTextureProvider) -> E2E:
        env = hermetic_env()
        blender = resolve_blender(blender_exe, env, BlenderSection())
        jobs: list[str] = []
        console = Console.recording()
        app = build_app(
            ctx_factory=production_factory(lib, env, blender, provider, jobs),
            console=console,
            env=env,
            cwd=lib,
        )
        return cls(lib, provider, blender, env, console, app, jobs=jobs)

    def gp(self, *args: str) -> tuple[Result, dict[str, Any]]:
        """``gp --json --blender <exe> <args> --library <lib>``; returns the parsed document."""
        argv = ["--json", "--blender", str(self.blender.path), *args, "--library", str(self.lib)]
        result = self.runner.invoke(self.app, argv, catch_exceptions=False)
        document: dict[str, Any] = json.loads(result.stdout)
        assert document["exit_code"] == result.exit_code
        return result, document

    def run(self, *extra: str) -> dict[str, Any]:
        """``gp run --model cube_rigged`` that must exit 0; returns the JSON document."""
        result, document = self.gp("run", "--model", MODEL, *extra)
        assert result.exit_code == 0, document
        return document

    # ------------------------------------------------------------------ on-disk artefacts
    @property
    def folder(self) -> Path:
        return self.lib / "models" / MODEL

    def state(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(
            (self.folder / ".pipeline" / "state.json").read_text(encoding="utf-8")
        )
        return data

    def stage_statuses(self) -> dict[str, str]:
        return {name: rec["status"] for name, rec in self.state()["stages"].items()}

    def work_json(self, rel: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.folder / rel).read_text(encoding="utf-8"))
        return data

    def manifest(self) -> dict[str, Any]:
        return self.work_json(f"output/{MODEL}/manifest.json")

    @staticmethod
    def report(document: dict[str, Any]) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(Path(document["report_path"]).read_text(encoding="utf-8"))
        return data

    @staticmethod
    def report_stages(document: dict[str, Any]) -> dict[str, str]:
        (model,) = E2E.report(document)["models"]
        assert model["model_id"] == MODEL
        return dict(model["stages"])

    def log_files(self) -> list[str]:
        logs = self.folder / ".pipeline" / "logs"
        return sorted(p.name for p in logs.iterdir()) if logs.is_dir() else []


def median_rgb(path: Path) -> tuple[int, ...]:
    """Per-channel median of an image's RGB pixels."""
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.int64).reshape(-1, 3)
    return tuple(int(v) for v in np.median(pixels, axis=0))
