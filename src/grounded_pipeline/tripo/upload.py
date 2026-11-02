"""File upload flows: multipart ``POST /files`` and presigned ``PUT`` (plan 10 §3; appendix A §5).

The HTTP mechanics (auth header, retries, envelope parsing, recording) belong to
:class:`~grounded_pipeline.tripo.client.TripoClient`; this module only decides *which* flow to
use and what content type / presign format a file needs. It talks to the client through the
narrow :class:`UploadHttp` protocol so it can be tested with a stub.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import structlog

from grounded_pipeline.domain.errors import TripoError
from grounded_pipeline.tripo.models import FileUploadData, PresignData

MIB = 1024 * 1024
MULTIPART_FIELD = "file"
OCTET_STREAM = "application/octet-stream"
EVENT_UPLOAD = "tripo.upload"

CONTENT_TYPES: Mapping[str, str] = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".fbx": OCTET_STREAM,
    ".obj": OCTET_STREAM,
    ".stl": OCTET_STREAM,
    ".3mf": OCTET_STREAM,
    ".usdz": OCTET_STREAM,
}
"""Extension → multipart ``Content-Type`` (unknown extensions fall back to octet-stream)."""

PRESIGN_FORMATS: frozenset[str] = frozenset(
    {"jpeg", "jpg", "png", "webp", "bmp", "tiff", "glb", "gltf", "fbx", "obj", "stl", "3mf", "usdz"}
)
"""``format`` values accepted by ``POST /files/presign`` (appendix A §1)."""

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class MultipartFile:
    """One file to send as a multipart form field; the client opens it per attempt."""

    field: str
    filename: str
    path: Path
    content_type: str


class UploadHttp(Protocol):
    """The two client operations the upload flows need."""

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
        """Authenticated API call with retries; returns the envelope's ``data``."""
        ...

    def put_unauthenticated(self, url: str, *, path: Path, content_type: str) -> None:
        """Stream ``path`` to ``url`` with ``PUT`` and no ``Authorization`` header."""
        ...


def content_type_for(path: Path) -> str:
    """Multipart content type for ``path`` by extension (``application/octet-stream`` default)."""
    return CONTENT_TYPES.get(path.suffix.lower(), OCTET_STREAM)


def presign_format_for(path: Path) -> str:
    """The ``format`` value for ``POST /files/presign`` derived from the file extension.

    Raises:
        TripoError: When the extension is not one Tripo accepts for presigned uploads.
    """
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in PRESIGN_FORMATS:
        raise TripoError(
            f"unsupported presign format {fmt!r} for {path.name}",
            detail={"path": str(path), "format": fmt, "supported": sorted(PRESIGN_FORMATS)},
            hint="Export the mesh as GLB or the image as PNG/JPEG before uploading.",
        )
    return fmt


def upload_file(http: UploadHttp, path: Path, *, presign_above_mb: int) -> str:
    """Upload ``path`` and return its ``file_token``.

    Files up to ``presign_above_mb`` MiB go through multipart ``POST /files``; larger ones use
    ``POST /files/presign`` followed by an unauthenticated ``PUT`` of the raw bytes.

    Args:
        http: The client (see :class:`UploadHttp`).
        path: Existing file to upload.
        presign_above_mb: Size threshold in MiB (``TripoSection.use_presign_above_mb``).

    Returns:
        The ``file_token`` to place in ``TextureTaskRequest.input`` or a ``texture_prompt``.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        TripoError: Any API/transport failure (see :mod:`grounded_pipeline.tripo.errors`).
    """
    size = path.stat().st_size
    if size > presign_above_mb * MIB:
        token = _upload_presigned(http, path)
        flow = "presign"
    else:
        token = _upload_multipart(http, path)
        flow = "multipart"
    _log.info(EVENT_UPLOAD, file=path.name, bytes=size, flow=flow)
    return token


def _upload_multipart(http: UploadHttp, path: Path) -> str:
    part = MultipartFile(
        field=MULTIPART_FIELD,
        filename=path.name,
        path=path,
        content_type=content_type_for(path),
    )
    data = http.request_envelope(
        "POST", "/files", FileUploadData, upload=part, context=f"POST /files ({path.name})"
    )
    return data.file_token


def _upload_presigned(http: UploadHttp, path: Path) -> str:
    fmt = presign_format_for(path)
    presign = http.request_envelope(
        "POST",
        "/files/presign",
        PresignData,
        json={"format": fmt},
        context=f"POST /files/presign ({path.name})",
    )
    http.put_unauthenticated(str(presign.presigned_url), path=path, content_type=OCTET_STREAM)
    return presign.file_token


__all__ = [
    "CONTENT_TYPES",
    "EVENT_UPLOAD",
    "MIB",
    "MULTIPART_FIELD",
    "OCTET_STREAM",
    "PRESIGN_FORMATS",
    "MultipartFile",
    "UploadHttp",
    "content_type_for",
    "presign_format_for",
    "upload_file",
]
