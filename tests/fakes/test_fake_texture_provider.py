"""Self-tests for ``FakeTextureProvider``: a broken fake poisons the stage and runner suites."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pygltflib
import pytest

from fakes.fake_texture_provider import (
    FakeTextureProvider,
    glb_copy_input,
    glb_multi_material_cube,
    glb_remeshed_cube,
    glb_reunwrapped_cube,
    glb_textured_cube,
    scaled_cube_factory,
)
from fakes.glb_builder import cube, write_glb
from tinyworld_pipeline.domain.errors import CancelledError, TripoTaskFailedError
from tinyworld_pipeline.domain.ports import CancelToken
from tinyworld_pipeline.texturing.provider import GuidanceMode, TextureProvider, TextureRequest


@pytest.fixture
def request_(tmp_path: Path) -> TextureRequest:
    mesh = write_glb(tmp_path / "work" / "static.glb", cube())
    return TextureRequest(
        model_id="ant",
        mesh_path=mesh,
        mode=GuidanceMode.TEXT,
        prompt="a red ant",
        style_image=None,
        guidance_images={},
        model_version="v3.5-20260815",
        quality="standard",
        pbr=True,
        bake=True,
        delight=True,
        alignment="original_image",
        seed=1,
        request_sha256="ab" * 32,
    )


def _load(path: Path) -> pygltflib.GLTF2:
    return pygltflib.GLTF2.load_from_bytes(path.read_bytes())


def _accessor(gltf: pygltflib.GLTF2, index: int) -> np.ndarray:
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()
    assert blob is not None
    start = view.byteOffset + acc.byteOffset
    comps = {"VEC3": 3, "VEC2": 2, "SCALAR": 1}[acc.type]
    raw = np.frombuffer(blob, dtype="<f4", count=acc.count * comps, offset=start)
    return raw.reshape(acc.count, comps)


def test_is_a_texture_provider(request_: TextureRequest) -> None:
    provider: TextureProvider = FakeTextureProvider()
    assert provider.estimate_credits(request_) == 10.0


def test_default_result_is_a_textured_glb_with_preview(request_: TextureRequest) -> None:
    fake = FakeTextureProvider(credits=7.0)
    res = fake.texture(request_, resume_task_id=None, cancel=CancelToken())
    assert res.provider == "tripo"
    assert res.task_id == "fake_task_0001"
    assert res.result_glb == request_.mesh_path.parent / "tripo" / "result.glb"
    assert res.preview_png == request_.mesh_path.parent / "tripo" / "preview.png"
    assert res.preview_png is not None
    assert res.preview_png.read_bytes().startswith(b"\x89PNG")
    assert res.credits_consumed == 7.0
    gltf = _load(res.result_glb)
    assert len(gltf.meshes[0].primitives) == 1
    assert len(gltf.materials) == 1
    assert len(gltf.images) == 5
    assert res.raw_output["status"] == "success"
    assert res.raw_output["credits_consumed"] == 7.0
    assert res.raw_output["request_body"]["texture_prompt"] == {"text": "a red ant"}
    assert "normalize_scale" not in res.raw_output["output"]


def test_records_calls_and_task_ids_deterministically(request_: TextureRequest) -> None:
    fake = FakeTextureProvider()
    fake.texture(request_, resume_task_id=None, cancel=CancelToken())
    fake.texture(request_, resume_task_id=None, cancel=CancelToken())
    res = fake.texture(request_, resume_task_id="task_resumed", cancel=CancelToken())
    assert fake.created_task_ids == ["fake_task_0001", "fake_task_0002"]
    assert [c.task_id for c in fake.calls] == ["fake_task_0001", "fake_task_0002", "task_resumed"]
    assert fake.last_call.resume_task_id == "task_resumed"
    assert fake.last_call.request is request_
    assert res.task_id == "task_resumed"
    assert res.raw_output["request_body"] is None
    assert fake.estimate_calls == []


def test_on_task_created_fires_before_return_and_before_error(request_: TextureRequest) -> None:
    events: list[str] = []
    fake = FakeTextureProvider()
    res = fake.texture(
        request_, resume_task_id=None, cancel=CancelToken(), on_task_created=events.append
    )
    events.append(f"returned:{res.task_id}")
    assert events == ["fake_task_0001", "returned:fake_task_0001"]

    failing = FakeTextureProvider(raise_=TripoTaskFailedError("boom"))
    seen: list[str] = []
    with pytest.raises(TripoTaskFailedError):
        failing.texture(
            request_, resume_task_id=None, cancel=CancelToken(), on_task_created=seen.append
        )
    assert seen == ["fake_task_0001"]


def test_raise_before_task_created_skips_the_callback(request_: TextureRequest) -> None:
    fake = FakeTextureProvider(raise_=TripoTaskFailedError("boom"), raise_before_task_created=True)
    seen: list[str] = []
    with pytest.raises(TripoTaskFailedError):
        fake.texture(
            request_, resume_task_id=None, cancel=CancelToken(), on_task_created=seen.append
        )
    assert seen == []
    assert len(fake.calls) == 1


def test_cancel_set_raises_cancelled(request_: TextureRequest) -> None:
    cancel = CancelToken()
    cancel.set()
    fake = FakeTextureProvider()
    with pytest.raises(CancelledError):
        fake.texture(request_, resume_task_id=None, cancel=cancel)
    assert fake.last_call.cancelled is True


def test_estimate_can_differ_from_credits(request_: TextureRequest) -> None:
    fake = FakeTextureProvider(credits=9.0, estimate=20.0)
    assert fake.estimate_credits(request_) == 20.0
    assert fake.texture(request_, resume_task_id=None, cancel=CancelToken()).credits_consumed == 9.0
    assert fake.estimate_calls == [request_]


def test_preview_false_writes_nothing(request_: TextureRequest) -> None:
    res = FakeTextureProvider(preview=False).texture(
        request_, resume_task_id=None, cancel=CancelToken()
    )
    assert res.preview_png is None
    assert not (request_.mesh_path.parent / "tripo" / "preview.png").exists()
    assert res.raw_output["output"]["rendered_image_url"] is None


def test_normalize_fields_reported_in_raw_output(request_: TextureRequest) -> None:
    fake = FakeTextureProvider(normalize_scale=0.5, normalize_translation=(0.0, -0.25, 0.0))
    res = fake.texture(request_, resume_task_id=None, cancel=CancelToken())
    assert res.raw_output["output"]["normalize_scale"] == 0.5
    assert res.raw_output["output"]["normalize_translation"] == [0.0, -0.25, 0.0]
    only_scale = FakeTextureProvider(normalize_scale=1.0)
    res2 = only_scale.texture(request_, resume_task_id=None, cancel=CancelToken())
    assert res2.raw_output["output"]["normalize_translation"] is None


# --------------------------------------------------------------------------- GLB factories
def test_copy_input_returns_identical_bytes(request_: TextureRequest) -> None:
    assert glb_copy_input(request_) == request_.mesh_path.read_bytes()


def test_reunwrapped_cube_has_same_positions_different_uvs(
    request_: TextureRequest, tmp_path: Path
) -> None:
    base = _load(write_glb(tmp_path / "base.glb", glb_textured_cube(request_)))
    other = _load(write_glb(tmp_path / "other.glb", glb_reunwrapped_cube(request_)))
    b_attrs, o_attrs = (
        base.meshes[0].primitives[0].attributes,
        other.meshes[0].primitives[0].attributes,
    )
    np.testing.assert_array_equal(
        _accessor(base, b_attrs.POSITION), _accessor(other, o_attrs.POSITION)
    )
    assert not np.array_equal(
        _accessor(base, b_attrs.TEXCOORD_0), _accessor(other, o_attrs.TEXCOORD_0)
    )


def test_remeshed_cube_changes_vertex_layout(request_: TextureRequest, tmp_path: Path) -> None:
    gltf = _load(write_glb(tmp_path / "remeshed.glb", glb_remeshed_cube(request_)))
    prim = gltf.meshes[0].primitives[0]
    assert gltf.accessors[prim.attributes.POSITION].count == 36  # 12 triangles x 3 split vertices
    assert len(gltf.materials) == 1


def test_multi_material_cube_has_two_primitives_and_materials(
    request_: TextureRequest, tmp_path: Path
) -> None:
    gltf = _load(write_glb(tmp_path / "multi.glb", glb_multi_material_cube(request_)))
    prims = gltf.meshes[0].primitives
    assert len(prims) == 2
    assert sorted(p.material for p in prims) == [0, 1]
    assert len(gltf.materials) == 2


def test_scaled_cube_factory_scales_positions(request_: TextureRequest, tmp_path: Path) -> None:
    fake = FakeTextureProvider(result_glb_factory=scaled_cube_factory(0.5), normalize_scale=0.5)
    res = fake.texture(request_, resume_task_id=None, cancel=CancelToken())
    gltf = _load(res.result_glb)
    positions = _accessor(gltf, gltf.meshes[0].primitives[0].attributes.POSITION)
    assert positions.max() == pytest.approx(0.25)
    assert positions.min() == pytest.approx(-0.25)
