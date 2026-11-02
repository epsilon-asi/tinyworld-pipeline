from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from grounded_pipeline.domain.errors import TripoError
from grounded_pipeline.tripo.models import FileUploadData, PresignData
from grounded_pipeline.tripo.upload import (
    MIB,
    MultipartFile,
    content_type_for,
    presign_format_for,
    upload_file,
)

from .conftest import API_KEY, BASE_URL, MakeClient, fixture_response, load_fixture

EnvelopeCall = tuple[str, str, Mapping[str, Any] | None, MultipartFile | None]


class StubHttp:
    """Records the calls made by ``upload_file`` and replays fixture data."""

    def __init__(self) -> None:
        self.envelope_calls: list[
            tuple[str, str, Mapping[str, Any] | None, MultipartFile | None]
        ] = []
        self.put_calls: list[tuple[str, Path, str]] = []

    def request_envelope[T](
        self,
        method: str,
        path: str,
        data_type: type[T],
        *,
        json: Mapping[str, Any] | None = None,
        upload: MultipartFile | None = None,
        context: str = "",
    ) -> T:
        self.envelope_calls.append((method, path, json, upload))
        name = "upload" if path == "/files" else "presign"
        result: T = data_type(**load_fixture(name)["data"])
        return result

    def put_unauthenticated(self, url: str, *, path: Path, content_type: str) -> None:
        self.put_calls.append((url, path, content_type))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mesh.glb", "model/gltf-binary"),
        ("mesh.GLB", "model/gltf-binary"),
        ("scene.gltf", "model/gltf+json"),
        ("ref.png", "image/png"),
        ("ref.jpg", "image/jpeg"),
        ("ref.jpeg", "image/jpeg"),
        ("ref.webp", "image/webp"),
        ("mesh.fbx", "application/octet-stream"),
        ("weird.xyz", "application/octet-stream"),
    ],
)
def test_content_type_by_extension(name: str, expected: str) -> None:
    assert content_type_for(Path(name)) == expected


def test_presign_format_from_extension() -> None:
    assert presign_format_for(Path("a.GLB")) == "glb"
    assert presign_format_for(Path("a.jpeg")) == "jpeg"
    with pytest.raises(TripoError, match="unsupported presign format") as info:
        presign_format_for(Path("a.blend"))
    assert info.value.detail["format"] == "blend"
    assert info.value.hint is not None


def test_small_file_uses_multipart(tmp_path: Path) -> None:
    mesh = tmp_path / "static.glb"
    mesh.write_bytes(b"glTF" * 10)
    http = StubHttp()
    token = upload_file(http, mesh, presign_above_mb=60)
    assert token == load_fixture("upload")["data"]["file_token"]
    assert http.put_calls == []
    (method, path, body, part) = http.envelope_calls[0]
    assert (method, path, body) == ("POST", "/files", None)
    assert part == MultipartFile(
        field="file", filename="static.glb", path=mesh, content_type="model/gltf-binary"
    )


def test_file_exactly_at_threshold_still_uses_multipart(tmp_path: Path) -> None:
    mesh = tmp_path / "static.glb"
    mesh.write_bytes(b"\0" * MIB)
    http = StubHttp()
    upload_file(http, mesh, presign_above_mb=1)
    assert [c[1] for c in http.envelope_calls] == ["/files"]


def test_large_file_uses_presign_then_unauthenticated_put(tmp_path: Path) -> None:
    mesh = tmp_path / "big.glb"
    mesh.write_bytes(b"\0" * (MIB + 1))
    http = StubHttp()
    token = upload_file(http, mesh, presign_above_mb=1)
    presign = load_fixture("presign")["data"]
    assert token == presign["file_token"]
    assert http.envelope_calls == [("POST", "/files/presign", {"format": "glb"}, None)]
    assert http.put_calls == [(presign["presigned_url"], mesh, "application/octet-stream")]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        upload_file(StubHttp(), tmp_path / "nope.glb", presign_above_mb=60)


# ------------------------------------------------------------------- end-to-end through the client
def test_client_multipart_upload_sends_file_field_and_auth(
    make_client: MakeClient, router: respx.MockRouter, tmp_path: Path
) -> None:
    route = router.post(f"{BASE_URL}/files").mock(return_value=fixture_response("upload"))
    mesh = tmp_path / "static.glb"
    mesh.write_bytes(b"glTF\x02\x00\x00\x00")
    client = make_client()
    token = client.upload(mesh, presign_above_mb=60)
    assert token == FileUploadData(**load_fixture("upload")["data"]).file_token
    request = route.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers["Content-Type"].startswith("multipart/form-data")
    body = request.read()
    assert b'name="file"; filename="static.glb"' in body
    assert b"Content-Type: model/gltf-binary" in body
    assert b"glTF\x02\x00\x00\x00" in body


def test_client_presign_upload_puts_without_auth(
    make_client: MakeClient, router: respx.MockRouter, tmp_path: Path
) -> None:
    presign = PresignData(**load_fixture("presign")["data"])
    presign_route = router.post(f"{BASE_URL}/files/presign").mock(
        return_value=fixture_response("presign")
    )
    put_route = router.put(url__eq=str(presign.presigned_url)).mock(
        return_value=httpx.Response(200)
    )
    payload = bytes(range(256)) * 4096 + b"tail"  # > 1 MiB, not chunk aligned
    mesh = tmp_path / "big.glb"
    mesh.write_bytes(payload)

    token = make_client().upload(mesh, presign_above_mb=1)

    assert token == presign.file_token
    assert presign_route.calls.last.request.read() == b'{"format":"glb"}'
    put = put_route.calls.last.request
    assert "authorization" not in {k.lower() for k in put.headers}
    assert put.headers["Content-Type"] == "application/octet-stream"
    assert put.headers["Content-Length"] == str(len(payload))
    assert put.read() == payload
