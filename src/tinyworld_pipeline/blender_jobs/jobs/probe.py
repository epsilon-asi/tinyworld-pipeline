"""``probe`` job: report Blender capabilities (plan 06 §5.1, appendix B §1-2).

Runs inside Blender's Python; ``bpy`` is imported lazily so the module loads under plain CPython.
"""

from __future__ import annotations

import platform
from typing import Any

from ..protocol import ProbeData, to_dict


def _operator_available(ops_module: Any, name: str) -> bool:  # noqa: ANN401 - bpy.ops namespace
    """True when ``bpy.ops.<module>.<name>`` resolves to a registered operator."""
    try:
        op = getattr(ops_module, name)
        op.get_rna_type()  # raises when the operator is not registered
    except Exception:
        return False
    return True


def _has_cycles(bpy: Any) -> bool:  # noqa: ANN401 - bpy module
    engines = {getattr(cls, "bl_idname", None) for cls in bpy.types.RenderEngine.__subclasses__()}
    if "CYCLES" in engines:
        return True
    return "cycles" in bpy.context.preferences.addons


def run(params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Collect :class:`ProbeData`; ``params`` are empty for this job."""
    del params
    import bpy  # noqa: PLC0415 - bpy only exists inside Blender

    warnings: list[str] = []
    fbx_legacy = _operator_available(bpy.ops.import_scene, "fbx")
    fbx_native = _operator_available(bpy.ops.wm, "fbx_import")
    if fbx_native and not fbx_legacy:
        warnings.append("Only the Blender 5.x native FBX importer is available; untested path.")
    data = ProbeData(
        blender_version=".".join(str(part) for part in bpy.app.version),
        python_version=platform.python_version(),
        has_cycles=_has_cycles(bpy),
        fbx_io_available=fbx_legacy or fbx_native,
        gltf_io_available=_operator_available(bpy.ops.import_scene, "gltf"),
    )
    result: dict[str, Any] = to_dict(data)
    return result, warnings
