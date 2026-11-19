"""Compare _no_lotus.png vs preview.png pixel-by-pixel to localize where
the lotus pads obscure visibility."""
import bpy, os, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "sRGB"
    w, h = int(img.size[0]), int(img.size[1])
    flat = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(flat)
    return flat.reshape((h, w, 4))[..., :3], w, h

a, _, _ = load(os.path.join(HERE, "test_scenes", "pabellon", "preview.png"))
b, _, _ = load(os.path.join(HERE, "test_scenes", "pabellon", "_no_lotus.png"))

# Save the difference (b - a) clipped to [0, 1] as a heatmap.
diff = np.clip(b - a, 0.0, 1.0)
img = bpy.data.images.new("diff", width=a.shape[1], height=a.shape[0],
                           alpha=False, float_buffer=True)
rgba = np.concatenate([diff, np.ones((a.shape[0], a.shape[1], 1), dtype=np.float32)], axis=-1)
img.pixels.foreach_set(rgba.reshape(-1))
out = os.path.join(HERE, "test_scenes", "pabellon", "_diff_lotus.png")
img.filepath_raw = out
img.file_format = "PNG"
sc = bpy.context.scene
sc.view_settings.view_transform = "Standard"
sc.render.image_settings.file_format = "PNG"
img.save_render(filepath=out)
print(f"wrote {out}")

# Also count black pixels in each
blackA = int((a.sum(-1) < 0.01).sum())
blackB = int((b.sum(-1) < 0.01).sum())
print(f"black pixels: with-lotus={blackA}, no-lotus={blackB}")
