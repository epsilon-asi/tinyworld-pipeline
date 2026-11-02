"""Extract a provider's textures from a GLB into the raw texture set (plan 11 §5.1).

The ``extract`` route is only valid when the provider kept our UVs; the ``apply`` stage decides
that from the verify report. This module copies the images of the single material the primitives
reference into ``raw/`` as PNG files (JPEG re-encoded losslessly to PNG) and records the material
factors so :mod:`grounded_pipeline.textures.urp_pack` can fold them into the URP maps.
"""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

from PIL import Image as PilImage
from PIL import UnidentifiedImageError

from grounded_pipeline.domain.errors import MultiMaterialResultError, UnsupportedGlbError
from grounded_pipeline.domain.textures import RawTextureSet, TextureFactors
from grounded_pipeline.gltf.loader import (
    GlbDocument,
    GlbImage,
    GlbMaterial,
    GlbTextureRef,
    sniff_image_mime,
)

RAW_FILENAMES: dict[str, str] = {
    "basecolor": "basecolor.png",
    "metallic_roughness": "metallic_roughness.png",
    "normal": "normal.png",
    "occlusion": "occlusion.png",
    "emissive": "emissive.png",
}


_DEFAULT_MATERIAL = GlbMaterial(name=None)


def extract_textures(doc: GlbDocument, out_dir: Path) -> RawTextureSet:
    """Write the referenced material's images to ``out_dir`` as PNGs.

    Args:
        doc: Loaded provider result.
        out_dir: Destination directory (created); files are named per :data:`RAW_FILENAMES`.

    Returns:
        The raw texture set with ``source="extract"``.

    Raises:
        MultiMaterialResultError: Primitives reference more than one distinct texture set
            (the ``apply`` stage falls back to the bake route on this error).
        UnsupportedGlbError: A referenced image cannot be decoded.
    """
    material = _single_material(doc)
    out_dir.mkdir(parents=True, exist_ok=True)
    slots: dict[str, GlbTextureRef | None] = {
        "basecolor": material.base_color_texture,
        "metallic_roughness": material.metallic_roughness_texture,
        "normal": material.normal_texture,
        "occlusion": material.occlusion_texture,
        "emissive": material.emissive_texture,
    }
    written: dict[str, Path | None] = {}
    for slot, ref in slots.items():
        if ref is None:
            written[slot] = None
            continue
        target = out_dir / RAW_FILENAMES[slot]
        target.write_bytes(_as_png(doc.images[ref.image_index], slot))
        written[slot] = target
    return RawTextureSet(
        basecolor=written["basecolor"],
        metallic_roughness=written["metallic_roughness"],
        normal=written["normal"],
        occlusion=written["occlusion"],
        emissive=written["emissive"],
        factors=TextureFactors(
            base_color=material.base_color_factor,
            metallic=material.metallic_factor,
            roughness=material.roughness_factor,
            emissive=material.emissive_factor,
            normal_scale=material.normal_scale,
            occlusion_strength=material.occlusion_strength,
            emissive_strength=material.emissive_strength,
        ),
        source="extract",
    )


def _single_material(doc: GlbDocument) -> GlbMaterial:
    """The one material all primitives share (names ignored), or the default when none is set."""
    referenced = [
        _DEFAULT_MATERIAL if m.material_index is None else doc.materials[m.material_index]
        for m in doc.meshes
    ]
    distinct = {replace(m, name=None) for m in referenced}
    if len(distinct) > 1:
        names = sorted({m.name or f"#{i}" for i, m in enumerate(doc.materials)})
        raise MultiMaterialResultError(
            f"result references {len(distinct)} distinct texture sets; cannot extract a single set",
            detail={"materials": names},
            hint="the apply stage bakes multi-material results onto the single prepared material",
        )
    return next(iter(distinct), _DEFAULT_MATERIAL)


def _as_png(image: GlbImage, slot: str) -> bytes:
    """PNG bytes for ``image``: verbatim when already PNG, else decoded and re-encoded."""
    if sniff_image_mime(image.data) == "image/png":
        return image.data
    try:
        with PilImage.open(io.BytesIO(image.data)) as decoded:
            buf = io.BytesIO()
            decoded.save(buf, format="PNG")
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedGlbError(
            f"{slot} image ({image.mime_type}) could not be decoded: {exc}", cause=exc
        ) from exc
    return buf.getvalue()
