"""Well-known paths inside an asset library and a model folder (plan 05 §1).

Nothing here touches the filesystem except :func:`find_library_root`, which only tests for the
existence of ``library.toml`` while walking up from a start directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

LIBRARY_TOML = "library.toml"
MODEL_MD = "model.md"
STYLE_MD = "style.md"
LIBRARY_ENV_VAR = "GP_LIBRARY"
DEFAULT_LIBRARY_DIRNAME = "library"


@dataclass(frozen=True, slots=True)
class ModelPaths:
    """Absolute paths of one ``models/<model_id>/`` folder."""

    model_id: str
    folder: Path

    @classmethod
    def for_model(cls, models_dir: Path, model_id: str) -> ModelPaths:
        """Build the paths for ``models_dir / model_id``."""
        return cls(model_id=model_id, folder=models_dir / model_id)

    @property
    def model_md(self) -> Path:
        return self.folder / MODEL_MD

    @property
    def source(self) -> Path:
        return self.folder / "source"

    @property
    def animations(self) -> Path:
        return self.source / "animations"

    @property
    def textures(self) -> Path:
        return self.source / "textures"

    @property
    def references(self) -> Path:
        return self.folder / "references"

    @property
    def work(self) -> Path:
        return self.folder / "work"

    @property
    def output(self) -> Path:
        return self.folder / "output"

    @property
    def pipeline(self) -> Path:
        return self.folder / ".pipeline"

    @property
    def state_file(self) -> Path:
        return self.pipeline / "state.json"

    @property
    def lock_file(self) -> Path:
        return self.pipeline / "state.lock"

    @property
    def logs(self) -> Path:
        return self.pipeline / "logs"


@dataclass(frozen=True, slots=True)
class LibraryPaths:
    """Absolute paths of the library-level files and folders."""

    root: Path

    @classmethod
    def from_root(cls, root: Path) -> LibraryPaths:
        """Build the paths for an (absolute-ised) library root."""
        return cls(root=root.expanduser().resolve())

    @property
    def library_toml(self) -> Path:
        return self.root / LIBRARY_TOML

    @property
    def style_dir(self) -> Path:
        return self.root / "style"

    @property
    def style_md(self) -> Path:
        return self.style_dir / STYLE_MD

    @property
    def style_references(self) -> Path:
        return self.style_dir / "references"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def generative_dir(self) -> Path:
        return self.root / "generative"

    @property
    def pipeline_dir(self) -> Path:
        return self.root / ".pipeline"

    @property
    def runs_dir(self) -> Path:
        return self.pipeline_dir / "runs"

    @property
    def run_lock(self) -> Path:
        return self.pipeline_dir / "run.lock"

    @property
    def gitignore(self) -> Path:
        return self.root / ".gitignore"

    def model(self, model_id: str) -> ModelPaths:
        """Paths of ``models/<model_id>/``."""
        return ModelPaths.for_model(self.models_dir, model_id)


def find_library_root(
    start: Path,
    env: Mapping[str, str],
    *,
    explicit: Path | None = None,
) -> Path | None:
    """Locate the asset library root (plan 08 global options).

    Precedence: ``explicit`` (the ``--library`` flag) > ``GP_LIBRARY`` > the nearest ancestor of
    ``start`` (inclusive) that contains ``library.toml`` > ``start/library`` if that directory
    exists. The flag and the environment variable are returned as given (resolved) even when the
    directory does not exist yet, so ``gp init --library new/`` can create it.

    Args:
        start: Directory to begin the ancestor walk from (usually the current working directory).
        env: Process environment (only ``GP_LIBRARY`` is consulted).
        explicit: Value of ``--library`` when the user passed it.

    Returns:
        The library root, or ``None`` when nothing matched.
    """
    if explicit is not None:
        return explicit.expanduser().resolve()
    from_env = env.get(LIBRARY_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser().resolve()
    start = start.expanduser().resolve()
    for candidate in (start, *start.parents):
        if (candidate / LIBRARY_TOML).is_file():
            return candidate
    default = start / DEFAULT_LIBRARY_DIRNAME
    if default.is_dir():
        return default
    return None
