"""``gp doctor`` with the fake Blender executable (probe round-trip through the real runner)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx

from fakes.fake_blender_exe import write_launcher
from tinyworld_pipeline.cli.commands import doctor as doctor_module
from tinyworld_pipeline.cli.render import Check
from tinyworld_pipeline.cli.wiring import resolve_blender

from .conftest import TEST_KEY, TRIPO_BASE, MakeHarness

BALANCE = {"code": 0, "data": {"balance": 42.0, "frozen": 0.0}}


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as mock:
        mock.get(f"{TRIPO_BASE}/account/balance").mock(
            return_value=httpx.Response(200, json=BALANCE)
        )
        yield mock


@pytest.fixture
def fake_blender(tmp_path: Path) -> Path:
    return write_launcher(tmp_path / "bin")


def _checks(document: dict[str, object]) -> dict[str, dict[str, object]]:
    checks = document["checks"]
    assert isinstance(checks, list)
    return {c["name"]: c for c in checks}


def test_doctor_all_pass_with_fake_blender(
    make_harness: MakeHarness,
    base_env: dict[str, str],
    fake_blender: Path,
    lib: Path,
    router: respx.MockRouter,
) -> None:
    harness = make_harness(env={**base_env, "TRIPO_API_KEY": TEST_KEY}, resolver=resolve_blender)
    result, document = harness.json("--blender", str(fake_blender), "doctor", "--library", str(lib))
    assert result.exit_code == 0, harness.text
    checks = _checks(document)
    assert checks["python"]["ok"] is True
    assert str(checks["blender"]["detail"]).startswith("4.5.12 at ")
    assert checks["blender version"]["ok"] is True
    assert checks["blender probe"]["ok"] is True
    assert "probe ok" in str(checks["blender probe"]["detail"])
    assert checks["tripo key"]["detail"] == "****cdef"
    assert checks["tripo balance"]["detail"] == "balance 42, frozen 0"
    assert checks["library"]["ok"] is True
    assert checks["free disk"]["ok"] is True
    assert "all checks passed" in harness.text
    assert "gp doctor" in harness.text


def test_doctor_without_key_and_library_exits_3(
    make_harness: MakeHarness, fake_blender: Path, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    harness = make_harness(resolver=resolve_blender, cwd=empty)
    result, document = harness.json("--blender", str(fake_blender), "doctor")
    assert result.exit_code == 3
    checks = _checks(document)
    assert checks["tripo key"]["ok"] is False
    assert checks["library"]["skipped"] is True
    assert checks["library"]["ok"] is True  # skipped is not a failure
    assert document["error"]["code"] == doctor_module.E_DOCTOR
    assert "tripo key" in document["error"]["message"]
    assert "1 check(s) failed" in harness.text


def test_doctor_unsupported_blender_version_fails(  # noqa: PLR0917 - fixtures
    make_harness: MakeHarness,
    base_env: dict[str, str],
    fake_blender: Path,
    lib: Path,
    router: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_BLENDER_VERSION", "5.0.0")
    harness = make_harness(env={**base_env, "TRIPO_API_KEY": TEST_KEY}, resolver=resolve_blender)
    result, document = harness.json("--blender", str(fake_blender), "doctor", "--library", str(lib))
    assert result.exit_code == 3
    checks = _checks(document)
    assert checks["blender"]["ok"] is False
    assert "not supported" in str(checks["blender"]["detail"])
    assert "blender probe" not in checks


def test_doctor_blender_not_found(make_harness: MakeHarness, lib: Path, tmp_path: Path) -> None:
    harness = make_harness(resolver=resolve_blender)
    result, document = harness.json(
        "--blender", str(tmp_path / "missing-blender"), "doctor", "--library", str(lib)
    )
    assert result.exit_code == 3
    assert _checks(document)["blender"]["ok"] is False


def test_doctor_invalid_library_and_balance_failure(
    make_harness: MakeHarness, base_env: dict[str, str], fake_blender: Path, lib: Path
) -> None:
    (lib / "models" / "cube_palette" / "source" / "cube_palette.fbx").unlink()
    with respx.mock(assert_all_mocked=True) as router:
        router.get(f"{TRIPO_BASE}/account/balance").mock(
            return_value=httpx.Response(500, json={"code": 2000, "message": "boom"})
        )
        harness = make_harness(
            env={**base_env, "TRIPO_API_KEY": TEST_KEY}, resolver=resolve_blender
        )
        result, document = harness.json(
            "--blender", str(fake_blender), "doctor", "--library", str(lib)
        )
    assert result.exit_code == 3
    checks = _checks(document)
    assert checks["library"]["ok"] is False
    assert "E_SOURCE_MISSING" in str(checks["library"]["detail"])
    assert checks["tripo balance"]["ok"] is False


def test_check_helpers() -> None:
    assert Check("x", True, "d").status == "pass"
    assert Check("x", False, "d").status == "FAIL"
    assert Check("x", True, "d", skipped=True).status == "skip"
    assert doctor_module.check_python().ok is True
    assert doctor_module.check_disk(Path.cwd()).name == "free disk"
