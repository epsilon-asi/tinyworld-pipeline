from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from grounded_pipeline.domain.ids import ModelId, RunId, Sha256, is_valid_model_id


@pytest.mark.parametrize("value", ["ant", "ant_worker", "a1", "x" * 64, "0abc"])
def test_is_valid_model_id_accepts_legal_names(value: str) -> None:
    assert is_valid_model_id(value)
    assert TypeAdapter(ModelId).validate_python(value) == value


@pytest.mark.parametrize(
    "value", ["", "A", "a", "_ant", "Ant", "ant-worker", "ant worker", "x" * 65]
)
def test_is_valid_model_id_rejects_illegal_names(value: str) -> None:
    assert not is_valid_model_id(value)
    with pytest.raises(ValidationError):
        TypeAdapter(ModelId).validate_python(value)


def test_run_id_is_ulid_shaped() -> None:
    TypeAdapter(RunId).validate_python("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    with pytest.raises(ValidationError):
        TypeAdapter(RunId).validate_python("01ARZ3NDEKTSV4RRFFQ69G5FA")  # 25 chars
    with pytest.raises(ValidationError):
        TypeAdapter(RunId).validate_python("01ARZ3NDEKTSV4RRFFQ69G5FAI")  # I not in Crockford


def test_sha256_is_lowercase_hex() -> None:
    TypeAdapter(Sha256).validate_python("a" * 64)
    with pytest.raises(ValidationError):
        TypeAdapter(Sha256).validate_python("A" * 64)
    with pytest.raises(ValidationError):
        TypeAdapter(Sha256).validate_python("a" * 63)
