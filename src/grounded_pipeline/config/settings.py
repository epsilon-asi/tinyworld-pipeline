"""Secrets (plan 07 §2). Only ``TRIPO_API_KEY`` in v1; never logged, never in TOML/front matter."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

USER_ENV_FILE = Path("~/.config/grounded-pipeline/.env")
KEY_SUFFIX_VISIBLE = 4


class Secrets(BaseSettings):
    """Loaded from the process environment, ``./.env`` and ``~/.config/grounded-pipeline/.env``."""

    model_config = SettingsConfigDict(
        env_file=(".env", str(USER_ENV_FILE)),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tripo_api_key: SecretStr | None = Field(None, validation_alias="TRIPO_API_KEY")

    @property
    def has_tripo_key(self) -> bool:
        return self.tripo_api_key is not None and bool(self.tripo_api_key.get_secret_value())

    def redacted_tripo_key(self) -> str:
        """``****abcd`` style rendering for ``gp doctor``; never the full value."""
        if not self.has_tripo_key:
            return "(not set)"
        assert self.tripo_api_key is not None  # for type narrowing
        value = self.tripo_api_key.get_secret_value()
        return "****" + value[-KEY_SUFFIX_VISIBLE:]


def load_secrets() -> Secrets:
    """Construct :class:`Secrets`, expanding ``~`` in the user env-file path."""
    files = (".env", str(USER_ENV_FILE.expanduser()))
    return Secrets(_env_file=files)
