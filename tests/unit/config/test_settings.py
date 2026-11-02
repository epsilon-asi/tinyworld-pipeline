from __future__ import annotations

from pathlib import Path

import pytest

from tinyworld_pipeline.config import settings as settings_mod
from tinyworld_pipeline.config.settings import Secrets, load_secrets


def test_key_from_environment_is_secret_and_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIPO_API_KEY", "sk-abcdefgh1234")

    secrets = Secrets(_env_file=None)

    assert secrets.has_tripo_key
    assert secrets.tripo_api_key is not None
    assert secrets.tripo_api_key.get_secret_value() == "sk-abcdefgh1234"
    assert "sk-abcdefgh1234" not in repr(secrets)
    assert secrets.redacted_tripo_key() == "****1234"


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRIPO_API_KEY", raising=False)
    secrets = Secrets(_env_file=None)
    assert not secrets.has_tripo_key
    assert secrets.redacted_tripo_key() == "(not set)"


def test_key_from_dot_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TRIPO_API_KEY", raising=False)
    (tmp_path / ".env").write_text("TRIPO_API_KEY=sk-fromfile\nOTHER=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_mod, "USER_ENV_FILE", tmp_path / "nope" / ".env")

    secrets = load_secrets()

    assert secrets.redacted_tripo_key() == "****file"


def test_environment_wins_over_dot_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TRIPO_API_KEY=sk-fromfile\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRIPO_API_KEY", "sk-fromenv")
    assert load_secrets().redacted_tripo_key() == "**** env".replace(" ", "m")[:4] + "menv"
