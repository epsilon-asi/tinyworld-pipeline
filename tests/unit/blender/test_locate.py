"""Plan 07 §4 / 04 §5: executable discovery order and version policy."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from fakes.fake_blender_exe import write_launcher
from grounded_pipeline.blender import locate
from grounded_pipeline.blender.locate import (
    ENV_EXECUTABLE,
    BlenderInfo,
    blender_info,
    blender_version,
    check_version,
    locate_blender,
    parse_blender_version,
)
from grounded_pipeline.domain.errors import BlenderNotFoundError, BlenderVersionError
from grounded_pipeline.domain.library_config import BlenderSection

MAC_APP = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def _exists_only(*paths: Path) -> Callable[[Path], bool]:
    present = {Path(p) for p in paths}
    return lambda candidate: Path(candidate) in present


def _no_glob(_pattern: str) -> list[str]:
    return []


def _locate(
    explicit: Path | None = None,
    env: dict[str, str] | None = None,
    config: BlenderSection | None = None,
    *,
    platform: str = "darwin",
    path_env: str | None = "",
    exists: Callable[[Path], bool] | None = None,
    glob: Callable[[str], list[str]] = _no_glob,
) -> Path:
    return locate_blender(
        explicit,
        env or {},
        config or BlenderSection(),
        platform=platform,
        path_env=path_env,
        exists=exists if exists is not None else _exists_only(),
        glob=glob,
    )


# --------------------------------------------------------------------------- search order
def test_explicit_flag_wins_over_everything() -> None:
    flag, env_exe, cfg_exe = Path("/flag/blender"), Path("/env/blender"), Path("/cfg/blender")
    found = _locate(
        flag,
        {ENV_EXECUTABLE: str(env_exe)},
        BlenderSection(executable=cfg_exe),
        path_env="/usr/bin",
        exists=_exists_only(flag, env_exe, cfg_exe, Path("/usr/bin/blender"), MAC_APP),
    )
    assert found == flag


def test_env_wins_over_config_and_path() -> None:
    env_exe, cfg_exe = Path("/env/blender"), Path("/cfg/blender")
    found = _locate(
        None,
        {ENV_EXECUTABLE: str(env_exe)},
        BlenderSection(executable=cfg_exe),
        path_env="/usr/bin",
        exists=_exists_only(env_exe, cfg_exe, Path("/usr/bin/blender")),
    )
    assert found == env_exe


def test_config_wins_over_path() -> None:
    cfg_exe = Path("/cfg/blender")
    found = _locate(
        config=BlenderSection(executable=cfg_exe),
        path_env="/usr/bin",
        exists=_exists_only(cfg_exe, Path("/usr/bin/blender")),
    )
    assert found == cfg_exe


def test_path_search_in_order_wins_over_well_known() -> None:
    found = _locate(
        path_env="/opt/none:/usr/local/bin:/usr/bin",
        exists=_exists_only(Path("/usr/local/bin/blender"), Path("/usr/bin/blender"), MAC_APP),
    )
    assert found == Path("/usr/local/bin/blender")


def test_path_env_defaults_to_env_PATH() -> None:  # noqa: N802
    found = _locate(
        env={"PATH": "/x/bin"}, path_env=None, exists=_exists_only(Path("/x/bin/blender"))
    )
    assert found == Path("/x/bin/blender")


def test_macos_well_known_system_then_user_applications() -> None:
    user_app = Path.home() / "Applications/Blender.app/Contents/MacOS/Blender"
    assert _locate(exists=_exists_only(MAC_APP, user_app)) == MAC_APP
    assert _locate(exists=_exists_only(user_app)) == user_app


def test_linux_well_known_locations_in_order() -> None:
    usr, snap = Path("/usr/bin/blender"), Path("/snap/bin/blender")
    opt = Path("/opt/blender-4.5.12/blender")
    opt_old = Path("/opt/blender-4.2.3/blender")
    flatpak = Path("/var/lib/flatpak/exports/bin/org.blender.Blender")

    def glob(pattern: str) -> list[str]:
        assert pattern == "/opt/blender*/blender"
        return [str(opt_old), str(opt)]

    assert _locate(platform="linux", exists=_exists_only(usr, snap, opt, flatpak)) == usr
    assert _locate(platform="linux", exists=_exists_only(snap, opt, flatpak), glob=glob) == snap
    # highest /opt version first
    assert _locate(platform="linux", exists=_exists_only(opt, opt_old, flatpak), glob=glob) == opt
    assert _locate(platform="linux", exists=_exists_only(opt_old, flatpak), glob=glob) == opt_old
    assert _locate(platform="linux", exists=_exists_only(flatpak)) == flatpak


def test_macos_search_does_not_consult_linux_paths() -> None:
    with pytest.raises(BlenderNotFoundError):
        _locate(platform="darwin", exists=_exists_only(Path("/usr/bin/blender")))


def test_unknown_platform_only_searches_path() -> None:
    with pytest.raises(BlenderNotFoundError):
        _locate(platform="freebsd14", exists=_exists_only(MAC_APP))
    assert _locate(
        platform="freebsd14",
        path_env="/usr/local/bin",
        exists=_exists_only(Path("/usr/local/bin/blender")),
    ) == Path("/usr/local/bin/blender")


def test_windows_platform_also_checks_blender_exe_on_path() -> None:
    exe = Path(r"C:\tools") / "blender.exe"
    found = _locate(platform="win32", path_env=r"C:\tools", exists=_exists_only(exe))
    assert found == exe


# --------------------------------------------------------------------------- failures
@pytest.mark.parametrize("source", ["flag", "env", "config"])
def test_configured_but_missing_path_raises_with_hint(source: str) -> None:
    missing = Path("/nowhere/blender")
    kwargs: dict[str, object] = {}
    if source == "flag":
        kwargs["explicit"] = missing
    elif source == "env":
        kwargs["env"] = {ENV_EXECUTABLE: str(missing)}
    else:
        kwargs["config"] = BlenderSection(executable=missing)
    with pytest.raises(BlenderNotFoundError) as info:
        _locate(**kwargs, exists=_exists_only(MAC_APP))  # type: ignore[arg-type]
    assert info.value.code == "E_BLENDER_NOT_FOUND"
    assert str(missing) in info.value.message
    assert info.value.hint
    assert info.value.detail["path"] == str(missing)


def test_nothing_found_lists_tried_paths_and_hint() -> None:
    with pytest.raises(BlenderNotFoundError) as info:
        _locate(path_env="/usr/bin")
    err = info.value
    assert err.exit_code == 2
    assert "/usr/bin/blender" in err.detail["tried"]
    assert str(MAC_APP) in err.detail["tried"]
    assert err.hint is not None
    assert ENV_EXECUTABLE in err.hint


def test_default_platform_and_exists_are_live() -> None:
    """Without injections the function consults the real filesystem of this host (smoke)."""
    if sys.platform == "darwin" and MAC_APP.exists():
        assert locate_blender(None, {}, BlenderSection(), path_env="") == MAC_APP
    else:
        with pytest.raises(BlenderNotFoundError):
            locate_blender(None, {}, BlenderSection(), path_env="")


# --------------------------------------------------------------------------- version parsing
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Blender 4.5.12\n\tbuild date: 2025-10-01", (4, 5, 12)),
        ("Blender 4.2.3 LTS", (4, 2, 3)),
        ("noise\nBlender 5.0.0\n", (5, 0, 0)),
    ],
)
def test_parse_blender_version(text: str, expected: tuple[int, int, int]) -> None:
    assert parse_blender_version(text) == expected


def test_parse_blender_version_rejects_garbage() -> None:
    with pytest.raises(BlenderVersionError) as info:
        parse_blender_version("Python 3.12.1")
    assert info.value.code == "E_BLENDER_VERSION"


# --------------------------------------------------------------------------- version policy
@pytest.mark.parametrize("version", [(4, 2, 3), (4, 2, 0), (4, 5, 12), (4, 5, 0)])
def test_supported_versions_produce_no_warnings(version: tuple[int, int, int]) -> None:
    assert check_version(version, allow_unsupported=False) == []


@pytest.mark.parametrize("version", [(4, 3, 2), (4, 4, 0)])
def test_tolerated_versions_warn(version: tuple[int, int, int]) -> None:
    warnings = check_version(version, allow_unsupported=False)
    assert len(warnings) == 1
    assert "not an LTS" in warnings[0]


@pytest.mark.parametrize("version", [(5, 2, 0), (5, 0, 0), (3, 6, 0), (4, 1, 1), (4, 6, 0)])
def test_unsupported_versions_raise_unless_allowed(version: tuple[int, int, int]) -> None:
    with pytest.raises(BlenderVersionError) as info:
        check_version(version, allow_unsupported=False)
    assert info.value.hint is not None
    assert "allow_unsupported" in info.value.hint

    warnings = check_version(version, allow_unsupported=True)
    assert len(warnings) == 1
    assert "allow_unsupported" in warnings[0]


# --------------------------------------------------------------------------- `blender --version`
def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["blender", "--version"], returncode=returncode, stdout=stdout, stderr=""
    )


def test_blender_version_runs_executable_with_version_flag() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return _completed("Blender 4.5.12\n")

    assert blender_version(Path("/x/blender"), run=run) == (4, 5, 12)
    assert calls == [["/x/blender", "--version"]]


def test_blender_version_maps_oserror_to_not_found() -> None:
    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(argv[0])

    with pytest.raises(BlenderNotFoundError):
        blender_version(Path("/x/blender"), run=run)


def test_blender_version_maps_timeout() -> None:
    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 60)

    with pytest.raises(BlenderVersionError):
        blender_version(Path("/x/blender"), run=run)


def test_blender_version_unparseable_output() -> None:
    def run(_argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed("", returncode=1)

    with pytest.raises(BlenderVersionError):
        blender_version(Path("/x/blender"), run=run)


def test_blender_info_dataclass() -> None:
    def run(_argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed("Blender 4.2.3")

    info = blender_info(Path("/x/blender"), run=run)
    assert info == BlenderInfo(path=Path("/x/blender"), version=(4, 2, 3))
    assert info.version_string == "4.2.3"


def test_blender_version_against_fake_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end with the fake exe (plan 15 §3.3): real subprocess, configurable version."""
    launcher = write_launcher(tmp_path)
    monkeypatch.setenv("FAKE_BLENDER_VERSION", "4.4.1")
    assert locate.blender_version(launcher) == (4, 4, 1)
