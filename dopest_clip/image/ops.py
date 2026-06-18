"""Local image operations (no provider, no network).

Built on Pillow (a base dependency). Two ops need optional deps and import them lazily,
raising a clear, actionable error if absent (no silent fallback):
  * remove_background -> rembg        (the `matting` extra)
  * svg_to_png        -> resvg-py     (the `graphics` extra)
  * png_to_svg        -> vtracer      (raster->vector tracer; if unavailable we raise
                                       NotImplementedError rather than fake a result)

Every op takes a `src` path and writes to `out`, returning {out, width, height, mode}
for raster results (get_image_info returns the probe dict; generate_icon_set returns the
list of written files). Colors are hex strings; "transparent"/"#0000" style alpha is
supported via _parse_color.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


# --- color parsing --------------------------------------------------------------------

def _parse_color(color):
    """Parse a hex color (#rgb/#rgba/#rrggbb/#rrggbbaa) or 'transparent' to an RGBA tuple.

    Pillow's ImageColor handles #rgb/#rrggbb/#rrggbbaa and names; we add 'transparent'
    and normalize everything to a 4-tuple so callers always get an alpha channel.
    """
    if color is None:
        return (0, 0, 0, 0)
    if isinstance(color, (tuple, list)):
        t = tuple(int(c) for c in color)
        if len(t) == 3:
            return t + (255,)
        if len(t) == 4:
            return t
        raise ValueError(f"color tuple must be length 3 or 4, got {t!r}")
    s = str(color).strip().lower()
    if s in ("transparent", "none", ""):
        return (0, 0, 0, 0)
    from PIL import ImageColor
    rgba = ImageColor.getcolor(s, "RGBA")
    return rgba


def _open_rgba(src) -> Image.Image:
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    return Image.open(p).convert("RGBA")


def _result(out: Path, img: Image.Image) -> dict:
    return {"out": str(out), "width": img.width, "height": img.height, "mode": img.mode}


def _save(img: Image.Image, out) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return _result(out, img)


# --- geometry ops ---------------------------------------------------------------------

def crop(src, out, x: int, y: int, w: int, h: int) -> dict:
    """Crop a w×h box whose top-left is (x, y). Raises if the box leaves the image."""
    img = _open_rgba(src)
    if w <= 0 or h <= 0:
        raise ValueError(f"crop w/h must be positive, got {w}x{h}")
    if x < 0 or y < 0 or x + w > img.width or y + h > img.height:
        raise ValueError(
            f"crop box ({x},{y},{w}x{h}) is outside image bounds {img.width}x{img.height}"
        )
    cropped = img.crop((x, y, x + w, y + h))
    return _save(cropped, out)


def pad(src, out, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0,
        color="#00000000") -> dict:
    """Add padding (px) on each side, filled with `color` (hex or 'transparent')."""
    img = _open_rgba(src)
    for n, v in (("left", left), ("top", top), ("right", right), ("bottom", bottom)):
        if v < 0:
            raise ValueError(f"pad {n} must be >= 0, got {v}")
    fill = _parse_color(color)
    new_w = img.width + left + right
    new_h = img.height + top + bottom
    canvas = Image.new("RGBA", (new_w, new_h), fill)
    canvas.alpha_composite(img, (left, top))
    return _save(canvas, out)


def resize(src, out, width: int | None = None, height: int | None = None,
           keep_aspect: bool = True) -> dict:
    """Resize to width/height. With keep_aspect (default), the missing dimension is
    derived; if both are given it fits inside the box preserving aspect. Without
    keep_aspect, both must be given and the image is stretched to exactly width×height.
    """
    img = _open_rgba(src)
    if width is None and height is None:
        raise ValueError("resize requires at least one of width/height")
    if not keep_aspect:
        if width is None or height is None:
            raise ValueError("resize with keep_aspect=False requires both width and height")
        target = (int(width), int(height))
        resized = img.resize(target, Image.LANCZOS)
        return _save(resized, out)
    # keep aspect
    if width is not None and height is not None:
        resized = img.copy()
        resized.thumbnail((int(width), int(height)), Image.LANCZOS)
    elif width is not None:
        ratio = int(width) / img.width
        resized = img.resize((int(width), max(1, round(img.height * ratio))), Image.LANCZOS)
    else:
        ratio = int(height) / img.height
        resized = img.resize((max(1, round(img.width * ratio)), int(height)), Image.LANCZOS)
    return _save(resized, out)


def square_canvas(src, out, size: int | None = None, bg="#00000000") -> dict:
    """Center the image on a square canvas of side `size` (defaults to the larger of the
    image's two dimensions), filled with `bg`. The source is not scaled, only centered."""
    img = _open_rgba(src)
    side = int(size) if size is not None else max(img.width, img.height)
    if side <= 0:
        raise ValueError(f"square size must be positive, got {side}")
    fill = _parse_color(bg)
    canvas = Image.new("RGBA", (side, side), fill)
    ox = (side - img.width) // 2
    oy = (side - img.height) // 2
    canvas.alpha_composite(img, (ox, oy))
    return _save(canvas, out)


# --- color ops ------------------------------------------------------------------------

def recolor(src, out, mode: str = "solid", to_hex: str = "#000000",
            from_hex: str | None = None, tolerance: int = 0) -> dict:
    """Recolor.

    modes:
      solid:     paint every non-transparent pixel with to_hex (alpha preserved)
      tint:      map grayscale luminance black->to_hex (preserves shading), alpha kept
      replace:   replace pixels matching from_hex (+/- tolerance) with to_hex
      grayscale: convert to grayscale (ignores to_hex)
    """
    img = _open_rgba(src)
    px = img.load()
    w, h = img.size

    if mode == "grayscale":
        gray = ImageOps.grayscale(img).convert("RGBA")
        # restore original alpha
        gray.putalpha(img.getchannel("A"))
        return _save(gray, out)

    if mode == "solid":
        r, g, b, _ = _parse_color(to_hex)
        for y in range(h):
            for x in range(w):
                _, _, _, a = px[x, y]
                if a > 0:
                    px[x, y] = (r, g, b, a)
        return _save(img, out)

    if mode == "tint":
        tr, tg, tb, _ = _parse_color(to_hex)
        for y in range(h):
            for x in range(w):
                cr, cg, cb, a = px[x, y]
                lum = (0.299 * cr + 0.587 * cg + 0.114 * cb) / 255.0
                px[x, y] = (round(tr * lum), round(tg * lum), round(tb * lum), a)
        return _save(img, out)

    if mode == "replace":
        if from_hex is None:
            raise ValueError("recolor mode 'replace' requires from_hex")
        fr, fg, fb, _ = _parse_color(from_hex)
        tr, tg, tb, _ = _parse_color(to_hex)
        tol = int(tolerance)
        for y in range(h):
            for x in range(w):
                cr, cg, cb, a = px[x, y]
                if abs(cr - fr) <= tol and abs(cg - fg) <= tol and abs(cb - fb) <= tol:
                    px[x, y] = (tr, tg, tb, a)
        return _save(img, out)

    raise ValueError(f"unknown recolor mode {mode!r}; expected solid|tint|replace|grayscale")


def invert_colors(src, out) -> dict:
    """Invert RGB channels, preserving alpha."""
    img = _open_rgba(src)
    r, g, b, a = img.split()
    rgb = Image.merge("RGB", (r, g, b))
    inverted = ImageOps.invert(rgb)
    out_img = Image.merge("RGBA", (*inverted.split(), a))
    return _save(out_img, out)


# --- background removal (lazy rembg) --------------------------------------------------

def remove_background(src, out) -> dict:
    """Remove the background, producing an RGBA cutout. Requires the `matting` extra
    (rembg). Raises a clear error if rembg is not installed."""
    try:
        from rembg import remove as _rembg_remove
    except ImportError as e:
        raise RuntimeError(
            "remove_background requires the 'rembg' package (install the matting extra: "
            "pip install 'dopest-clip[matting]'). It is not installed."
        ) from e
    img = _open_rgba(src)
    cutout = _rembg_remove(img)
    if not isinstance(cutout, Image.Image):
        cutout = Image.open(cutout) if not isinstance(cutout, (bytes, bytearray)) else \
            Image.open(__import__("io").BytesIO(cutout))
    cutout = cutout.convert("RGBA")
    return _save(cutout, out)


# --- svg <-> png ----------------------------------------------------------------------

def svg_to_png(src, out, width: int | None = None, height: int | None = None) -> dict:
    """Rasterize an SVG to PNG. Requires the `graphics` extra (resvg-py). Raises a clear
    error if resvg is not installed."""
    try:
        import resvg_py
    except ImportError as e:
        raise RuntimeError(
            "svg_to_png requires the 'resvg-py' package (install the graphics extra: "
            "pip install 'dopest-clip[graphics]'). It is not installed."
        ) from e
    src_p = Path(src)
    if not src_p.exists():
        raise FileNotFoundError(f"svg not found: {src_p}")
    svg_text = src_p.read_text(encoding="utf-8")
    kwargs = {}
    if width is not None:
        kwargs["width"] = int(width)
    if height is not None:
        kwargs["height"] = int(height)
    # resvg_py.svg_to_bytes returns PNG bytes (or a list of ints depending on version).
    png = resvg_py.svg_to_bytes(svg_string=svg_text, **kwargs)
    if isinstance(png, list):
        png = bytes(png)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    with Image.open(out) as img:
        return _result(out, img)


def png_to_svg(src, out, **opts) -> dict:
    """Trace a raster image to SVG.

    Pillow has no vector-tracing capability, and there is no pure-Python tracer in the
    declared dependency set. If the optional `vtracer` package is installed we use it;
    otherwise we raise NotImplementedError (we do NOT fake a result by embedding the
    raster as a base64 <image>, which would not be a real trace)."""
    try:
        import vtracer
    except ImportError as e:
        raise NotImplementedError(
            "png_to_svg (raster->vector tracing) has no pure-Python implementation in "
            "the dopest-clip dependency set. Install the 'vtracer' package to enable it "
            "(pip install vtracer). Faking a trace by embedding the bitmap is not done."
        ) from e
    src_p = Path(src)
    if not src_p.exists():
        raise FileNotFoundError(f"image not found: {src_p}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vtracer.convert_image_to_svg_py(str(src_p), str(out), **opts)
    return {"out": str(out)}


# --- probe / icon set -----------------------------------------------------------------

def get_image_info(src) -> dict:
    """Return {width, height, mode, format} without converting the image."""
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    with Image.open(p) as img:
        return {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format,
        }


def generate_icon_set(src, out_dir, sizes=None, name: str = "icon",
                      square: bool = True, bg="#00000000") -> dict:
    """Write {name}-{size}.png for each size into out_dir, downscaling from the source.

    The source is centered on a square canvas first (when square=True) so non-square
    inputs scale without distortion. Returns {out_dir, files:[...], sizes:[...]}.
    """
    if sizes is None:
        sizes = [16, 32, 48, 64, 128, 180, 192, 256, 512]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _open_rgba(src)
    if square and base.width != base.height:
        side = max(base.width, base.height)
        canvas = Image.new("RGBA", (side, side), _parse_color(bg))
        canvas.alpha_composite(base, ((side - base.width) // 2, (side - base.height) // 2))
        base = canvas

    files = []
    for size in sizes:
        size = int(size)
        if size <= 0:
            raise ValueError(f"icon size must be positive, got {size}")
        icon = base.resize((size, size), Image.LANCZOS)
        dest = out_dir / f"{name}-{size}.png"
        icon.save(dest)
        files.append(str(dest))

    return {"out_dir": str(out_dir), "files": files, "sizes": [int(s) for s in sizes]}
