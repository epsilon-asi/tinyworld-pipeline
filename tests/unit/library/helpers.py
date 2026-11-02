"""Programmatic library builders and helpers for the ``library/*`` unit tests (not fixtures).

Every invalid-library scenario is built in ``tmp_path`` from a valid baseline, so each test states
exactly the one thing it breaks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
from PIL import Image

from tinyworld_pipeline.config.layering import CliOverrides
from tinyworld_pipeline.domain.errors import Problem
from tinyworld_pipeline.library.discovery import Library
from tinyworld_pipeline.library.validation import BlenderCheck, validate_library

FIXTURE_LIBRARY = Path(__file__).resolve().parents[2] / "fixtures" / "library_minimal"

DEFAULT_TOML = 'schema_version = 1\n\n[library]\nname = "test"\n'

DEFAULT_STYLE_MD = """---
schema_version: 1
name: test-style
---

## base_prompt
Stylized hand-painted game asset, soft shading.

## material_notes
Matte paint, soft specular.

## palette
#FF0000 red, #00FF00 green
"""

FBX_PLACEHOLDER = b"Kaydara FBX Binary  \x00placeholder\n"


def write_png(path: Path, size: tuple[int, int] = (256, 256)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (200, 100, 50)).save(path, format="PNG")
    return path


def write_jpeg(path: Path, size: tuple[int, int] = (256, 256)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (20, 120, 220)).save(path, format="JPEG")
    return path


def model_md_text(front: Mapping[str, Any], body: str) -> str:
    text: str = frontmatter.dumps(frontmatter.Post(body, **dict(front)), sort_keys=False)
    return text + "\n"


@dataclass
class LibraryBuilder:
    """A valid baseline library under ``root`` that tests then mutate."""

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.toml(DEFAULT_TOML)
        self.style(DEFAULT_STYLE_MD)
        write_png(self.root / "style" / "references" / "primary.png")

    # -- files
    def write(self, rel: str, text: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_bytes(self, rel: str, data: bytes) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def toml(self, text: str) -> Path:
        return self.write("library.toml", text)

    def style(self, text: str) -> Path:
        return self.write("style/style.md", text)

    def model(
        self,
        model_id: str = "cube",
        front: Mapping[str, Any] | None = None,
        *,
        body: str = "## prompt\nA red cube with brass corners.\n",
        fbx: tuple[str, ...] = ("source/cube.fbx",),
        raw: str | None = None,
    ) -> Path:
        """Create ``models/<model_id>/`` with a ``model.md`` and placeholder FBX files."""
        folder = self.root / "models" / model_id
        folder.mkdir(parents=True, exist_ok=True)
        for rel in fbx:
            self.write_bytes(f"models/{model_id}/{rel}", FBX_PLACEHOLDER)
        if raw is None:
            raw = model_md_text({"display_name": "Cube", **(front or {})}, body)
        (folder / "model.md").write_text(raw, encoding="utf-8")
        return folder

    def load(self) -> Library:
        return Library.load(self.root)


def validate(
    lib: Library,
    ids: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cli: CliOverrides | None = None,
    blender_check: BlenderCheck | None = None,
) -> list[Problem]:
    return validate_library(
        lib, ids, env=env or {}, cli=cli or CliOverrides(), blender_check=blender_check
    )


def codes(problems: list[Problem]) -> list[str]:
    return [p.code for p in problems]


def errors(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity == "error"]
