"""Image subsystem tests — run in the light venv (pytest, pytest-mock, Pillow, requests,
mcp). Pillow IS present; rembg/resvg/vtracer are NOT; no network.

Local ops are exercised for real (Pillow only). Optional-dep ops (remove_background,
svg_to_png, png_to_svg) are skipped when their dep is absent. gen.py is tested by
monkeypatching registry.get("image") to a fake provider returning known bytes.
"""

from __future__ import annotations

import importlib.util

import pytest
from PIL import Image

from dopest_clip import project
from dopest_clip.image import gen, ops
from dopest_clip.providers import registry


# --- helpers ---------------------------------------------------------------------------

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_image(path, size=(40, 30), color=(200, 100, 50, 255)):
    img = Image.new("RGBA", size, color)
    img.save(path)
    return path


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


needs_rembg = pytest.mark.skipif(not _have("rembg"), reason="rembg (matting extra) not installed")
needs_resvg = pytest.mark.skipif(not _have("resvg_py"), reason="resvg-py (graphics extra) not installed")
needs_vtracer = pytest.mark.skipif(not _have("vtracer"), reason="vtracer not installed")


@pytest.fixture
def src_image(tmp_path):
    return _make_image(tmp_path / "src.png")


# --- local ops: geometry ---------------------------------------------------------------

def test_crop(src_image, tmp_path):
    out = tmp_path / "crop.png"
    res = ops.crop(src_image, out, 5, 5, 10, 8)
    assert res["width"] == 10 and res["height"] == 8
    with Image.open(out) as img:
        assert img.size == (10, 8)


def test_crop_out_of_bounds(src_image, tmp_path):
    with pytest.raises(ValueError):
        ops.crop(src_image, tmp_path / "c.png", 0, 0, 999, 999)


def test_pad(src_image, tmp_path):
    out = tmp_path / "pad.png"
    res = ops.pad(src_image, out, left=2, top=3, right=4, bottom=5, color="#ffffffff")
    assert res["width"] == 40 + 2 + 4
    assert res["height"] == 30 + 3 + 5
    with Image.open(out) as img:
        # padded corner should be the fill color (white, opaque)
        assert img.convert("RGBA").getpixel((0, 0)) == (255, 255, 255, 255)


def test_resize_keep_aspect_width_only(src_image, tmp_path):
    out = tmp_path / "r.png"
    res = ops.resize(src_image, out, width=20)
    assert res["width"] == 20
    # 40x30 -> width 20 -> height 15
    assert res["height"] == 15


def test_resize_stretch(src_image, tmp_path):
    out = tmp_path / "r.png"
    res = ops.resize(src_image, out, width=10, height=10, keep_aspect=False)
    assert (res["width"], res["height"]) == (10, 10)


def test_resize_requires_dim(src_image, tmp_path):
    with pytest.raises(ValueError):
        ops.resize(src_image, tmp_path / "r.png")


def test_square_canvas_default(src_image, tmp_path):
    out = tmp_path / "sq.png"
    res = ops.square_canvas(src_image, out)
    assert res["width"] == res["height"] == 40  # max(40,30)


def test_square_canvas_sized(src_image, tmp_path):
    out = tmp_path / "sq.png"
    res = ops.square_canvas(src_image, out, size=64, bg="transparent")
    assert res["width"] == res["height"] == 64
    with Image.open(out) as img:
        assert img.convert("RGBA").getpixel((0, 0))[3] == 0  # transparent corner


# --- local ops: color ------------------------------------------------------------------

def test_invert_colors(tmp_path):
    src = _make_image(tmp_path / "s.png", size=(4, 4), color=(10, 20, 30, 200))
    out = tmp_path / "inv.png"
    ops.invert_colors(src, out)
    with Image.open(out) as img:
        px = img.convert("RGBA").getpixel((0, 0))
        assert px == (245, 235, 225, 200)  # 255-10,255-20,255-30, alpha kept


def test_recolor_solid(tmp_path):
    src = _make_image(tmp_path / "s.png", size=(4, 4), color=(10, 20, 30, 255))
    out = tmp_path / "rc.png"
    ops.recolor(src, out, mode="solid", to_hex="#ff0000")
    with Image.open(out) as img:
        assert img.convert("RGBA").getpixel((0, 0)) == (255, 0, 0, 255)


def test_recolor_grayscale(tmp_path):
    src = _make_image(tmp_path / "s.png", size=(4, 4), color=(255, 0, 0, 255))
    out = tmp_path / "g.png"
    ops.recolor(src, out, mode="grayscale")
    with Image.open(out) as img:
        r, g, b, a = img.convert("RGBA").getpixel((0, 0))
        assert r == g == b and a == 255


def test_recolor_replace_requires_from(src_image, tmp_path):
    with pytest.raises(ValueError):
        ops.recolor(src_image, tmp_path / "x.png", mode="replace", to_hex="#000000")


# --- local ops: probe + icon set -------------------------------------------------------

def test_get_image_info(src_image):
    info = ops.get_image_info(src_image)
    assert info["width"] == 40 and info["height"] == 30
    assert info["mode"] in ("RGBA", "RGB")
    assert info["format"] == "PNG"


def test_get_image_info_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ops.get_image_info(tmp_path / "nope.png")


def test_generate_icon_set(src_image, tmp_path):
    out_dir = tmp_path / "icons"
    res = ops.generate_icon_set(src_image, out_dir, sizes=[16, 32, 64])
    assert res["sizes"] == [16, 32, 64]
    assert len(res["files"]) == 3
    for size, f in zip([16, 32, 64], res["files"]):
        with Image.open(f) as img:
            assert img.size == (size, size)


# --- optional-dep ops (skip when dep absent) -------------------------------------------

@needs_rembg
def test_remove_background(src_image, tmp_path):
    out = tmp_path / "cut.png"
    res = ops.remove_background(src_image, out)
    assert res["mode"] == "RGBA"


def test_remove_background_clear_error_when_missing(src_image, tmp_path, monkeypatch):
    if _have("rembg"):
        pytest.skip("rembg installed; missing-dep path not exercised")
    with pytest.raises(RuntimeError) as ei:
        ops.remove_background(src_image, tmp_path / "x.png")
    assert "rembg" in str(ei.value)


@needs_resvg
def test_svg_to_png(tmp_path):
    svg = tmp_path / "s.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
        '<rect width="20" height="20" fill="#ff0000"/></svg>',
        encoding="utf-8",
    )
    out = tmp_path / "s.png"
    res = ops.svg_to_png(svg, out)
    assert res["out"] == str(out)
    with Image.open(out) as img:
        assert img.width > 0


def test_svg_to_png_clear_error_when_missing(tmp_path):
    if _have("resvg_py"):
        pytest.skip("resvg-py installed; missing-dep path not exercised")
    svg = tmp_path / "s.svg"
    svg.write_text("<svg/>", encoding="utf-8")
    with pytest.raises(RuntimeError) as ei:
        ops.svg_to_png(svg, tmp_path / "o.png")
    assert "resvg" in str(ei.value)


def test_png_to_svg_not_implemented_when_missing(src_image, tmp_path):
    if _have("vtracer"):
        pytest.skip("vtracer installed; NotImplementedError path not exercised")
    with pytest.raises(NotImplementedError):
        ops.png_to_svg(src_image, tmp_path / "o.svg")


# --- gen.py: provider-routed (fake provider, no network) -------------------------------

class FakeImageProvider:
    name = "fake"

    def __init__(self):
        self.calls = []

    def generate_image(self, prompt, model, **opts):
        self.calls.append(("generate", prompt, model, opts))
        return PNG_MAGIC + b"GEN"

    def edit_image(self, image_bytes, instruction, model, **opts):
        self.calls.append(("edit", instruction, model, opts))
        return PNG_MAGIC + b"EDIT"

    def compose_images(self, image_bytes_list, instruction, model, **opts):
        self.calls.append(("compose", instruction, model, len(image_bytes_list)))
        return PNG_MAGIC + b"COMPOSE"

    def analyze_image(self, image_bytes, instruction, model, **opts):
        self.calls.append(("analyze", instruction, model, opts))
        return {"text": "a fake description"}


@pytest.fixture
def fake_provider(monkeypatch):
    fp = FakeImageProvider()
    monkeypatch.setattr(registry, "get", lambda cap: fp if cap == "image" else (_ for _ in ()).throw(AssertionError(cap)))
    return fp


def _new_project(projects_root, pid="img-proj"):
    project.ensure_project(pid)
    project.write_meta(pid, {"project_id": pid})
    return pid


def test_generate_persists_to_project(fake_provider, projects_root):
    pid = _new_project(projects_root)
    res = gen.generate("a red cube", model="gemini-2.5-flash-image", project_id=pid)
    assert res["provider"] == "fake"
    assert res["model"] == "gemini-2.5-flash-image"
    out = project.image_out_path(pid, "generated", "png")
    assert str(out) == res["out"]
    assert out.read_bytes() == PNG_MAGIC + b"GEN"


def test_generate_explicit_out(fake_provider, tmp_path):
    out = tmp_path / "sub" / "thing.png"
    res = gen.generate("x", model="m", out=out)
    assert res["out"] == str(out)
    assert out.read_bytes() == PNG_MAGIC + b"GEN"


def test_generate_requires_model(fake_provider, tmp_path):
    with pytest.raises(ValueError) as ei:
        gen.generate("x", model="", out=tmp_path / "o.png")
    assert "model" in str(ei.value).lower()


def test_generate_requires_output_location(fake_provider):
    with pytest.raises(ValueError) as ei:
        gen.generate("x", model="m")
    assert "out" in str(ei.value) or "project_id" in str(ei.value)


def test_edit_reads_and_persists(fake_provider, tmp_path):
    src = _make_image(tmp_path / "src.png")
    out = tmp_path / "edited.png"
    res = gen.edit(src, "make it blue", model="m", out=out)
    assert out.read_bytes() == PNG_MAGIC + b"EDIT"
    assert fake_provider.calls[0][0] == "edit"


def test_edit_missing_input(fake_provider, tmp_path):
    with pytest.raises(FileNotFoundError):
        gen.edit(tmp_path / "nope.png", "x", model="m", out=tmp_path / "o.png")


def test_compose(fake_provider, tmp_path):
    a = _make_image(tmp_path / "a.png")
    b = _make_image(tmp_path / "b.png")
    out = tmp_path / "c.png"
    res = gen.compose([a, b], "merge", model="m", out=out)
    assert out.read_bytes() == PNG_MAGIC + b"COMPOSE"
    assert fake_provider.calls[0] == ("compose", "merge", "m", 2)


def test_compose_requires_paths(fake_provider, tmp_path):
    with pytest.raises(ValueError):
        gen.compose([], "x", model="m", out=tmp_path / "o.png")


def test_analyze(fake_provider, tmp_path):
    src = _make_image(tmp_path / "src.png")
    res = gen.analyze(src, "what is this?", model="m")
    assert res == {"text": "a fake description"}


def test_analyze_requires_model(fake_provider, tmp_path):
    src = _make_image(tmp_path / "src.png")
    with pytest.raises(ValueError):
        gen.analyze(src, "x", model="")
