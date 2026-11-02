"""Library discovery: enumerate models, parse documents, resolve source files (plan 05, 09 §1).

:class:`Library` is a lazily-parsing view over a library root. Loading is *lenient*: a broken
``library.toml`` or ``style.md`` is recorded in :attr:`Library.load_problems` instead of raising, so
``gp validate`` can report it next to the per-model findings. Callers that need a usable library
(``resolve_model``) raise on those problems first.
"""

from __future__ import annotations

from pathlib import Path

from tinyworld_pipeline.config.layering import load_library_toml
from tinyworld_pipeline.domain.errors import ConfigError, Problem, UsageError, ValidationError
from tinyworld_pipeline.domain.ids import is_valid_model_id
from tinyworld_pipeline.domain.library_config import LibraryConfig
from tinyworld_pipeline.domain.model_spec import ModelDocument
from tinyworld_pipeline.domain.style_spec import StyleDocument
from tinyworld_pipeline.library.frontmatter_io import E_FRONTMATTER, parse_model_md, parse_style_md
from tinyworld_pipeline.library.paths import LibraryPaths

E_LIBRARY_TOML = "E_LIBRARY_TOML"
E_SOURCE_MISSING = "E_SOURCE_MISSING"
E_SOURCE_AMBIGUOUS = "E_SOURCE_AMBIGUOUS"
E_ANIM_GLOB_EMPTY = "E_ANIM_GLOB_EMPTY"

FBX_SUFFIX = ".fbx"


def _is_fbx(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == FBX_SUFFIX


# --------------------------------------------------------------------------- source resolution
def find_primary_fbx(doc: ModelDocument) -> tuple[Path | None, list[Problem]]:
    """Resolve the primary source FBX of a model (plan 05 §2).

    Explicit ``source.primary`` (relative to the model folder) wins; otherwise exactly one
    top-level ``.fbx`` in ``source/`` is required.

    Args:
        doc: Parsed ``model.md``.

    Returns:
        ``(absolute path or None, problems)`` — ``E_SOURCE_MISSING`` or ``E_SOURCE_AMBIGUOUS``.
    """
    explicit = doc.front.source.primary
    if explicit is not None:
        candidate = (doc.folder / explicit).resolve()
        if _is_fbx(candidate):
            return candidate, []
        return None, [
            Problem(
                code=E_SOURCE_MISSING,
                severity="error",
                model_id=doc.model_id,
                path="model.md:source.primary",
                message=f"source.primary {explicit.as_posix()!r} is not an existing .fbx file",
                hint="fix the path or drop source.primary to auto-detect a single source/*.fbx",
            )
        ]
    source_dir = doc.folder / "source"
    found = sorted(p for p in source_dir.glob("*") if _is_fbx(p)) if source_dir.is_dir() else []
    if len(found) == 1:
        return found[0].resolve(), []
    if not found:
        return None, [
            Problem(
                code=E_SOURCE_MISSING,
                severity="error",
                model_id=doc.model_id,
                path="source/",
                message="no .fbx file found at the top level of source/",
                hint="copy the rigged FBX into source/ or set source.primary",
            )
        ]
    names = ", ".join(p.name for p in found)
    return None, [
        Problem(
            code=E_SOURCE_AMBIGUOUS,
            severity="error",
            model_id=doc.model_id,
            path="source/",
            message=f"several top-level .fbx files in source/ ({names}) and no source.primary",
            hint="set source.primary: source/<file>.fbx in model.md",
        )
    ]


def find_animation_fbx(
    doc: ModelDocument, *, primary: Path | None = None
) -> tuple[list[Path], list[Problem]]:
    """Expand ``source.animations`` globs (relative to the model folder), sorted and de-duplicated.

    An explicit glob that matches nothing is an ``E_ANIM_GLOB_EMPTY`` error; when the list is the
    default (``source.animations`` not set in front matter) the same code is emitted as a warning.

    Args:
        doc: Parsed ``model.md``.
        primary: The primary FBX, excluded from the result if a glob happens to match it.

    Returns:
        ``(sorted absolute paths, problems)``.
    """
    explicit = "animations" in doc.front.source.model_fields_set
    matches: set[Path] = set()
    problems: list[Problem] = []
    for pattern in doc.front.source.animations:
        hits = [p.resolve() for p in doc.folder.glob(pattern) if _is_fbx(p)]
        hits = [p for p in hits if primary is None or p != primary]
        if not hits:
            problems.append(
                Problem(
                    code=E_ANIM_GLOB_EMPTY,
                    severity="error" if explicit else "warning",
                    model_id=doc.model_id,
                    path="model.md:source.animations",
                    message=f"animation pattern {pattern!r} matches no .fbx file",
                    hint="drop the pattern or add clips under source/animations/",
                )
            )
        matches.update(hits)
    return sorted(matches), problems


# --------------------------------------------------------------------------- Library
class Library:
    """A loaded asset library: paths, ``library.toml`` config, style document, model folders."""

    def __init__(
        self,
        paths: LibraryPaths,
        config: LibraryConfig,
        style: StyleDocument | None,
        load_problems: list[Problem],
    ) -> None:
        self.paths = paths
        self.config = config
        self.style = style
        self.load_problems = list(load_problems)
        self._docs: dict[str, ModelDocument] = {}

    @classmethod
    def load(cls, root: Path) -> Library:
        """Load ``library.toml`` and ``style/style.md`` from ``root`` (leniently, see module doc).

        Args:
            root: Library root directory.

        Returns:
            A :class:`Library`; inspect :attr:`load_problems` for ``E_LIBRARY_TOML`` /
            ``E_FRONTMATTER`` findings. A broken ``library.toml`` falls back to defaults.
        """
        paths = LibraryPaths.from_root(root)
        problems: list[Problem] = []
        config = LibraryConfig()
        try:
            config = load_library_toml(paths.library_toml)
        except ConfigError as exc:
            problems.append(
                Problem(
                    code=E_LIBRARY_TOML,
                    severity="error",
                    path="library.toml",
                    message=exc.message,
                    hint="run `gp init --force` to regenerate a commented default library.toml",
                )
            )
        style: StyleDocument | None = None
        if paths.style_md.is_file():
            try:
                style = parse_style_md(paths.style_md)
            except ValidationError as exc:
                problems.extend(exc.problems)
        else:
            problems.append(
                Problem(
                    code=E_FRONTMATTER,
                    severity="error",
                    path="style/style.md",
                    message="style/style.md not found",
                    hint="run `gp init` to scaffold it",
                )
            )
        return cls(paths, config, style, problems)

    # ----------------------------------------------------------------- enumeration
    def model_ids(self) -> list[str]:
        """Sorted names of every folder under ``models/`` (including invalid names, so that
        validation can flag ``E_MODEL_ID``). Hidden folders are skipped."""
        models_dir = self.paths.models_dir
        if not models_dir.is_dir():
            return []
        return sorted(
            p.name for p in models_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def select(self, ids: list[str] | None = None, tags: list[str] | None = None) -> list[str]:
        """Choose models by id and/or tag (plan 08 §2 ``gp run`` selection).

        Args:
            ids: Explicit model ids; an unknown id raises before any work.
            tags: Models carrying at least one of these tags (requires parsing ``model.md``).

        Returns:
            Sorted model ids: the union of both criteria, or every model when neither is given.

        Raises:
            UsageError: An id in ``ids`` has no folder.
            ValidationError: A ``model.md`` needed for tag filtering fails to parse.
        """
        known = self.model_ids()
        if not ids and not tags:
            return known
        chosen: set[str] = set()
        for model_id in ids or []:
            if model_id not in known:
                raise UsageError(
                    f"unknown model id {model_id!r}",
                    hint="known ids: " + (", ".join(known) or "(none)"),
                )
            chosen.add(model_id)
        if tags:
            wanted = set(tags)
            for model_id in known:
                if is_valid_model_id(model_id) and wanted & set(
                    self.model_document(model_id).front.tags
                ):
                    chosen.add(model_id)
        return sorted(chosen)

    # ----------------------------------------------------------------- documents
    def model_document(self, model_id: str) -> ModelDocument:
        """Parse (and cache) ``models/<model_id>/model.md``.

        Raises:
            UsageError: Invalid id or no such folder.
            ValidationError: ``E_MODEL_MD_MISSING``, ``E_FRONTMATTER`` or ``E_UNKNOWN_KEY``.
        """
        if model_id in self._docs:
            return self._docs[model_id]
        paths = self.paths.model(model_id)
        if not is_valid_model_id(model_id) or not paths.folder.is_dir():
            raise UsageError(f"unknown model id {model_id!r}")
        if not paths.model_md.is_file():
            raise ValidationError(
                [
                    Problem(
                        code="E_MODEL_MD_MISSING",
                        severity="error",
                        model_id=model_id,
                        path="model.md",
                        message="model.md not found",
                        hint="run `gp new-model` or create model.md by hand (plan 05 §5)",
                    )
                ]
            )
        doc = parse_model_md(paths.model_md, model_id)
        self._docs[model_id] = doc
        return doc
