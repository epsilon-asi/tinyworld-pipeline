"""Blender executable discovery and version policy (plan 07 §4, 04 §5).

Resolution order: explicit ``--blender`` flag → ``GP_BLENDER__EXECUTABLE`` →
``[blender].executable`` in ``library.toml`` → ``PATH`` → well-known install locations for the
current platform. Every filesystem probe goes through an injectable ``exists``/``glob`` so the
search is unit-testable on any host. Writing the ``library/.pipeline/blender.json`` cache is the
CLI's job (WP-11); this module only produces :class:`BlenderInfo`.
"""

from __future__ import annotations

import glob as _glob
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinyworld_pipeline.domain.errors import BlenderNotFoundError, BlenderVersionError
from tinyworld_pipeline.domain.library_config import BlenderSection

ENV_EXECUTABLE = "GP_BLENDER__EXECUTABLE"
"""Environment variable naming the Blender executable (pydantic-settings nested delimiter form)."""

_VERSION_RE = re.compile(r"Blender (\d+)\.(\d+)\.(\d+)")
_VERSION_TIMEOUT_S = 60

SUPPORTED_MINORS: frozenset[int] = frozenset({2, 5})
"""Blender 4.x minors that are fully supported (LTS releases)."""
TOLERATED_MINORS: frozenset[int] = frozenset({3, 4})
"""Blender 4.x minors that run with a warning (non-LTS)."""

Version = tuple[int, int, int]
_Runner = Callable[..., "subprocess.CompletedProcess[str]"]
_Exists = Callable[[Path], bool]
_Glob = Callable[[str], Iterable[str]]


@dataclass(frozen=True)
class BlenderInfo:
    """Resolved executable and its parsed version (what ``gp doctor`` shows and WP-11 caches)."""

    path: Path
    version: Version

    @property
    def version_string(self) -> str:
        """Dotted version, e.g. ``"4.5.12"``."""
        return ".".join(str(part) for part in self.version)


# --------------------------------------------------------------------------- discovery
def _path_candidates(path_env: str, names: tuple[str, ...], *, windows: bool) -> list[Path]:
    out: list[Path] = []
    for entry in path_env.split(";" if windows else ":"):
        if not entry:
            continue
        out.extend(Path(entry) / name for name in names)
    return out


def _well_known(platform: str, env: Mapping[str, str], glob: _Glob) -> list[Path]:
    """Well-known install locations for ``platform`` (plan 07 §4)."""
    if platform == "darwin":
        rel = Path("Applications/Blender.app/Contents/MacOS/Blender")
        return [Path("/") / rel, Path.home() / rel]
    if platform.startswith("linux"):
        fixed = [Path("/usr/bin/blender"), Path("/snap/bin/blender")]
        opt = sorted((Path(p) for p in glob("/opt/blender*/blender")), reverse=True)
        flatpak = [
            Path("/var/lib/flatpak/exports/bin/org.blender.Blender"),
            Path.home() / ".local/share/flatpak/exports/bin/org.blender.Blender",
        ]
        return [*fixed, *opt, *flatpak]
    if platform.startswith("win"):  # pragma: no cover - Windows host deferred (ADR-0002)
        program_files = env.get("ProgramFiles", r"C:\Program Files")
        versioned = sorted(
            (Path(p) for p in glob(rf"{program_files}\Blender Foundation\Blender 4.*\blender.exe")),
            reverse=True,
        )
        steam = [
            Path(program_files + r" (x86)") / r"Steam\steamapps\common\Blender\blender.exe",
            Path(program_files) / r"Steam\steamapps\common\Blender\blender.exe",
        ]
        store = []
        if local := env.get("LOCALAPPDATA"):
            store.append(Path(local) / r"Microsoft\WindowsApps\blender.exe")
        return [*versioned, *steam, *store]
    return []


def locate_blender(
    explicit: Path | None,
    env: Mapping[str, str],
    config: BlenderSection,
    *,
    platform: str = sys.platform,
    path_env: str | None = None,
    exists: _Exists = Path.exists,
    glob: _Glob = _glob.glob,
) -> Path:
    """Return the Blender executable to use, following the plan 07 §4 search order.

    Args:
        explicit: Value of the ``--blender`` flag, if given.
        env: Process environment (``GP_BLENDER__EXECUTABLE`` is honoured).
        config: ``[blender]`` section of ``library.toml``.
        platform: ``sys.platform`` value; injected for tests.
        path_env: ``PATH`` to search; defaults to ``env["PATH"]``.
        exists: Filesystem probe; injected for tests.
        glob: Glob function for wildcard install locations; injected for tests.

    Returns:
        The first existing candidate.

    Raises:
        BlenderNotFoundError: An explicitly configured path does not exist, or no candidate exists.
    """
    windows = platform.startswith("win")
    names = ("blender.exe", "blender") if windows else ("blender",)
    tried: list[str] = []

    configured: list[tuple[str, Path]] = []
    if explicit is not None:
        configured.append(("--blender", explicit))
    if env_value := env.get(ENV_EXECUTABLE):
        configured.append((ENV_EXECUTABLE, Path(env_value)))
    if config.executable is not None:
        configured.append(("[blender].executable", config.executable))
    for source, candidate in configured:
        path = candidate.expanduser()
        if exists(path):
            return path
        raise BlenderNotFoundError(
            f"Blender executable from {source} does not exist: {path}",
            detail={"source": source, "path": str(path)},
            hint="Fix the path or remove the setting to fall back to auto-discovery.",
        )

    search = path_env if path_env is not None else env.get("PATH", "")
    for candidate in [
        *_path_candidates(search, names, windows=windows),
        *_well_known(platform, env, glob),
    ]:
        if exists(candidate):
            return candidate
        tried.append(str(candidate))

    raise BlenderNotFoundError(
        "No Blender executable found.",
        detail={"tried": tried, "platform": platform},
        hint=(
            "Install Blender 4.5 LTS from blender.org, then pass --blender <path>, set "
            f"{ENV_EXECUTABLE}, or set [blender].executable in library.toml."
        ),
    )


# --------------------------------------------------------------------------- version
def parse_blender_version(text: str) -> Version:
    """Extract ``(major, minor, patch)`` from ``blender --version`` output.

    Raises:
        BlenderVersionError: No ``Blender X.Y.Z`` token is present.
    """
    match = _VERSION_RE.search(text)
    if match is None:
        raise BlenderVersionError(
            "Could not parse Blender version from `blender --version` output.",
            detail={"output": text[:500]},
            hint="Check that the configured executable is really Blender.",
        )
    major, minor, patch = (int(group) for group in match.groups())
    return (major, minor, patch)


def check_version(version: Version, *, allow_unsupported: bool) -> list[str]:
    """Apply the support policy from plan 04 §5.

    4.2.x and 4.5.x are supported silently; 4.3.x/4.4.x warn; anything else (< 4.2, 4.6+, ≥ 5.0)
    is refused unless ``allow_unsupported`` turns the refusal into a warning.

    Returns:
        Warnings to surface to the user (empty when fully supported).

    Raises:
        BlenderVersionError: Unsupported version and ``allow_unsupported`` is false.
    """
    major, minor, _ = version
    label = ".".join(str(part) for part in version)
    if major == 4 and minor in SUPPORTED_MINORS:  # noqa: PLR2004 - the 4.x series
        return []
    if major == 4 and minor in TOLERATED_MINORS:  # noqa: PLR2004
        return [f"Blender {label} is not an LTS release; 4.5 LTS is recommended."]
    message = f"Blender {label} is not supported (supported: 4.5.x LTS, 4.2.x LTS)."
    if allow_unsupported:
        return [f"{message} Continuing because blender.allow_unsupported = true."]
    raise BlenderVersionError(
        message,
        detail={"version": label},
        hint=(
            "Install Blender 4.5 LTS from blender.org (Homebrew casks ship 5.x), or set "
            "[blender].allow_unsupported = true to proceed at your own risk."
        ),
    )


def blender_version(exe: Path, run: _Runner = subprocess.run) -> Version:
    """Run ``exe --version`` and parse the result.

    Args:
        exe: Blender executable.
        run: ``subprocess.run``-compatible callable; injected for tests.

    Raises:
        BlenderNotFoundError: The executable could not be started.
        BlenderVersionError: It did not print a parseable version.
    """
    argv = [str(exe), "--version"]
    try:
        completed: subprocess.CompletedProcess[str] = run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=_VERSION_TIMEOUT_S,
        )
    except OSError as exc:
        raise BlenderNotFoundError(
            f"Cannot execute Blender at {exe}: {exc}",
            detail={"path": str(exe)},
            hint="Check the path and that the file is executable.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise BlenderVersionError(
            f"`{exe} --version` timed out after {_VERSION_TIMEOUT_S}s.",
            detail={"path": str(exe)},
        ) from exc
    output: Any = completed.stdout or ""
    return parse_blender_version(str(output))


def blender_info(exe: Path, run: _Runner = subprocess.run) -> BlenderInfo:
    """Convenience: resolve the version of ``exe`` into a :class:`BlenderInfo`."""
    return BlenderInfo(path=exe, version=blender_version(exe, run))
