"""Image buffers and legacy texture baking shared by material and graph export."""

from __future__ import annotations

import time

from . import native_scene
from .export_context import ExportContext


class _PreBakedTexture:
    """A synthetic linear RGB texture produced during legacy export (e.g. by
    pre-mixing the two sides of a two-texture Mix node).

    Carries `rgb` as a numpy (h, w, 3) float32 array and a `cache_key` so the
    main exporter can dedupe and write it without going through `bpy.Image`.
    """
    __slots__ = ("rgb", "w", "h", "cache_key")

    def __init__(self, rgb, w: int, h: int, cache_key: str):
        self.rgb = rgb
        self.w = w
        self.h = h
        self.cache_key = cache_key


def _read_image_pixels(image, *, context=None):
    """Read a fresh RGBA buffer; missing images use Cycles' magenta sentinel."""
    if context is None:
        context = ExportContext()
    import numpy as np
    try:
        width, height = map(int, image.size)
        if width == 0 or height == 0:
            image.update()
            width, height = map(int, image.size)
        if width <= 0 or height <= 0:
            raise ValueError("empty image")
        pixels = np.empty(width * height * 4, np.float32)
        started = time.perf_counter()
        image.pixels.foreach_get(pixels)
        context.stats["pixel_read_s"] += time.perf_counter() - started
        context.stats["pixel_bytes"] += pixels.nbytes
        return width, height, pixels
    except (RuntimeError, ValueError) as exc:
        context.warn(f"missing-image:{image.name}",
              f"image {image.name!r} has no readable pixels ({exc}); using magenta")
        return 1, 1, np.array([1.0, 0.0, 1.0, 1.0], np.float32)


def export_image_texture(
    image,
    writer,
    textures: list,
    colorspace: str | None = None,
    chain: tuple = (),
    *, context=None,
) -> int:
    """Retain an image's float32 pixel buffer and register its TextureDesc.

    Accepts either a `bpy.types.Image` (chain transforms baked into its pixels)
    or a `_PreBakedTexture` (already in linear space; chain is applied on top of
    the prebaked pixels). Returns the texture index. Reuses existing entries by
    cache key.
    """
    if context is None:
        context = ExportContext()
    import numpy as np
    if isinstance(image, _PreBakedTexture):
        return _export_prebaked(image, writer, textures, chain=chain, context=context)
    chain_key = "" if not chain else "|" + repr(tuple(x[0] for x in chain))
    key = f"__image__{image.name}{chain_key}"
    for i, t in enumerate(textures):
        if t.get("_key") == key:
            return i
    t_enter = time.perf_counter()
    pixel_before = context.stats["pixel_read_s"]
    bake_before = context.stats["bake_chain_s"]
    width, height, pixels = _read_image_pixels(image, context=context)
    if colorspace is None:
        colorspace = (
            "srgb"
            if image.colorspace_settings.name.lower().startswith("srgb")
            else "linear"
        )
    if chain:
        pixels, colorspace = _bake_chain(pixels, width, height, colorspace, chain, context=context)
    if context.texture_pct is not None and context.texture_pct != 100:
        new_w = max(1, int(round(width * context.texture_pct / 100.0)))
        new_h = max(1, int(round(height * context.texture_pct / 100.0)))
        if (new_w, new_h) != (width, height):
            rgba = pixels.reshape((height, width, 4))
            rgba = _resample_bilinear(rgba, new_h, new_w)
            pixels = np.ascontiguousarray(rgba, dtype=np.float32).reshape(-1)
            width, height = new_w, new_h
    desc = {
        "width": int(width),
        "height": int(height),
        "channels": 4,
        "colorspace": colorspace,
        "_key": key,
        # Add the index into the writer's separate texture-array list.
        **writer.write_texture_pixels(pixels),
    }
    textures.append(desc)
    dt = time.perf_counter() - t_enter
    children = (context.stats["pixel_read_s"] - pixel_before) + (context.stats["bake_chain_s"] - bake_before)
    context.stats["texture_self_s"] += max(0.0, dt - children)
    context.stats["texture_count"] += 1
    if dt > 0.25:
        slow = context.stats["slow_textures"]
        slow.append((image.name, int(width), int(height), dt))
        slow.sort(key=lambda e: e[3], reverse=True)
        del slow[5:]
    return len(textures) - 1


def _export_prebaked(pb: "_PreBakedTexture", writer, textures: list, chain: tuple = (), *, context=None) -> int:
    if context is None:
        context = ExportContext()
    import numpy as np
    # A non-empty chain carries the residual colour transforms that sit above
    # the baked Mix (e.g. a terminal HueSaturation Value dimming). It must be
    # folded into the cache key so two materials sharing the same prebaked
    # base but a different terminal transform get distinct entries.
    chain_key = "" if not chain else "|" + repr(tuple(x[0] for x in chain))
    key = pb.cache_key + chain_key
    for i, t in enumerate(textures):
        if t.get("_key") == key:
            return i
    rgb = pb.rgb
    w, h = int(pb.w), int(pb.h)
    if chain:
        # Prebaked pixels are already linear; apply the chain transforms in
        # place (texture-first → outer-last, same order as `_bake_chain`).
        # Done at full resolution before the texture_pct resample, mirroring
        # the bpy.Image path in `export_image_texture`.
        rgb = np.asarray(rgb, dtype=np.float32).reshape((h, w, 3)).copy()
        for _id, apply_fn in chain:
            rgb = apply_fn(rgb)
        rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    if context.texture_pct is not None and context.texture_pct != 100:
        new_w = max(1, int(round(w * context.texture_pct / 100.0)))
        new_h = max(1, int(round(h * context.texture_pct / 100.0)))
        if (new_w, new_h) != (w, h):
            rgb = _resample_bilinear(rgb, new_h, new_w)
            w, h = new_w, new_h
    rgba = np.concatenate(
        [rgb, np.ones((h, w, 1), dtype=np.float32)], axis=-1
    )
    pixels = np.ascontiguousarray(rgba, dtype=np.float32).reshape(-1)
    desc = {
        "width": w,
        "height": h,
        "channels": 4,
        "colorspace": "linear",
        "_key": key,
        **writer.write_texture_pixels(pixels),
    }
    textures.append(desc)
    return len(textures) - 1


def _image_to_linear_rgb(image, *, context=None):
    """Decode a `bpy.types.Image` into a (h, w, 3) numpy array in linear space."""
    if context is None:
        context = ExportContext()
    import numpy as np
    ptr = image.as_pointer()
    hit = context.linear_rgb_cache.get(ptr)
    if hit is not None:
        context.stats["linear_rgb_cache_hits"] += 1
        return hit
    context.stats["linear_rgb_cache_misses"] += 1
    w, h, px = _read_image_pixels(image, context=context)
    px = px.reshape((h, w, 4))
    rgb = np.ascontiguousarray(px[..., :3], dtype=np.float32)
    if image.colorspace_settings.name.lower().startswith("srgb"):
        rgb = _srgb_to_linear_np(rgb)
    # Freeze so downstream chain / resample steps can't accidentally mutate the
    # shared buffer. All known apply_fn/resample/mix helpers return new arrays.
    rgb.setflags(write=False)
    result = (rgb, w, h)
    context.linear_rgb_cache[ptr] = result
    return result


def _bake_source_to_linear(source, chain, *, context=None):
    """Resolve any texture source (`bpy.Image` or `_PreBakedTexture`) plus chain
    into a linear (h, w, 3) numpy array.
    """
    if context is None:
        context = ExportContext()
    if isinstance(source, _PreBakedTexture):
        # PreBaked is already linear; chains on top would be unusual.
        rgb = source.rgb
        for _id, apply_fn in chain:
            rgb = apply_fn(rgb)
        return rgb, source.w, source.h
    rgb, w, h = _image_to_linear_rgb(source, context=context)
    for _id, apply_fn in chain:
        rgb = apply_fn(rgb)
    return rgb, w, h


def _resample_bilinear(rgb, target_h: int, target_w: int):
    """Separable bilinear resample — X pass first, then Y. The 4-corner form
    allocated 4 × (target_h, target_w, c) intermediates per call; classroom's
    700 → 2048 upsample made that ~200MB and ~300ms. Separable drops the
    heaviest pass to (h, target_w, c).
    """
    import numpy as np
    h, w, _ = rgb.shape
    if h == target_h and w == target_w:
        return rgb
    native = native_scene.accelerator()
    if native is not None and rgb.dtype == np.float32:
        return native.resample_bilinear(np.ascontiguousarray(rgb), target_h, target_w)
    # X pass: interpolate columns, shape (h, target_w, c).
    xs = np.linspace(0.0, w - 1, target_w, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    fx = (xs - x0)[None, :, None]
    a = rgb[:, x0, :]
    b = rgb[:, x1, :]
    row_interp = a + (b - a) * fx
    # Y pass: interpolate rows, shape (target_h, target_w, c).
    ys = np.linspace(0.0, h - 1, target_h, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    fy = (ys - y0)[:, None, None]
    c0 = row_interp[y0]
    c1 = row_interp[y1]
    out = c0 + (c1 - c0) * fy
    return out.astype(np.float32, copy=False)


def _srgb_to_linear_np(x):
    """sRGB → linear conversion, kept in float32 end-to-end.

    The previous `np.power(..., 2.4)` with a Python-float exponent silently
    promoted the whole image to f64 and `np.where` over f32+f64 operands
    produced an f64 intermediate, so a 2K image would allocate ~50MB of f64
    it didn't need. Using an explicit `np.float32(2.4)` + `dtype=np.float32`
    keeps memory and arithmetic at half that.
    """
    import numpy as np
    xf = x if x.dtype == np.float32 else x.astype(np.float32)
    native = native_scene.accelerator()
    if native is not None:
        return native.srgb_to_linear(np.ascontiguousarray(xf))
    lin = xf * np.float32(1.0 / 12.92)
    gamma = np.power(
        (xf + np.float32(0.055)) * np.float32(1.0 / 1.055),
        np.float32(2.4),
        dtype=np.float32,
    )
    return np.where(xf <= np.float32(0.04045), lin, gamma)


def _bake_chain(pixels, w, h, colorspace, chain, *, context=None):
    """Apply `chain` color transforms to RGBA f32 pixels.

    Transforms operate in linear space (matching Cycles' shader semantics: the
    TexImage node converts sRGB→linear before its Color output). If the source
    is sRGB, we linearise before applying. Returns the new pixel list and the
    output colorspace string ("linear" once anything is baked in).
    """
    if context is None:
        context = ExportContext()
    import numpy as np
    t_enter = time.perf_counter()
    try:
        arr = np.asarray(pixels, dtype=np.float32).reshape((h, w, 4)).copy()
        rgb = arr[..., :3].copy()
        if colorspace.lower() == "srgb":
            rgb = _srgb_to_linear_np(rgb).astype(np.float32)
        for _id, apply_fn in chain:
            rgb = apply_fn(rgb)
        arr[..., :3] = rgb
        # The writer accepts NumPy arrays directly, without a list round-trip.
        return arr.reshape(-1), "linear"
    finally:
        context.stats["bake_chain_s"] += time.perf_counter() - t_enter
