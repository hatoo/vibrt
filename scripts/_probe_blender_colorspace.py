"""Test Blender's `Image.pixels.foreach_get` behaviour for the same JPG
under different colorspace tags. Goal: figure out whether
foreach_get returns linear-decoded or raw values, so we know how
vibrt's exporter should handle Non-Color tags.
"""
import bpy
import numpy as np

# Pick the church brick diff JPG (currently Non-Color) and read its
# pixels. Then flip its colorspace tag, re-read, compare.
img = bpy.data.images["castle_brick_02_red_diff_1k.jpg"]
print(f"image: {img.filepath}")

# Original tag
orig_cs = img.colorspace_settings.name
print(f"\noriginal colorspace = {orig_cs!r}")
w, h = img.size
pixels = np.empty(w * h * 4, dtype=np.float32)
img.pixels.foreach_get(pixels)
arr = pixels.reshape((h, w, 4))[..., :3]
center_pix = arr[h//2, w//2, :]
print(f"  center pixel: ({center_pix[0]:.4f}, {center_pix[1]:.4f}, {center_pix[2]:.4f})")
print(f"  mean: ({arr[..., 0].mean():.4f}, {arr[..., 1].mean():.4f}, {arr[..., 2].mean():.4f})")

# Now flip to sRGB and re-read
img.colorspace_settings.name = "sRGB"
# Force reload via image.update? Actually pixels cache may not refresh.
# Best: clear pixels access by reload
img.reload()
img.update()
new_cs = img.colorspace_settings.name
print(f"\nafter setting colorspace = {new_cs!r}")
pixels2 = np.empty(w * h * 4, dtype=np.float32)
img.pixels.foreach_get(pixels2)
arr2 = pixels2.reshape((h, w, 4))[..., :3]
center_pix2 = arr2[h//2, w//2, :]
print(f"  center pixel: ({center_pix2[0]:.4f}, {center_pix2[1]:.4f}, {center_pix2[2]:.4f})")
print(f"  mean: ({arr2[..., 0].mean():.4f}, {arr2[..., 1].mean():.4f}, {arr2[..., 2].mean():.4f})")

# Now flip back
img.colorspace_settings.name = orig_cs
img.reload()
img.update()

# Compare: if both readings differ, Blender DOES apply colorspace transform
# on read. If same, Blender returns raw pixels.
diff = arr2 - arr
print(f"\n=== sRGB - Non-Color difference ===")
print(f"  max |diff|: {np.abs(diff).max():.5f}")
print(f"  mean |diff|: {np.abs(diff).mean():.5f}")
print(f"  center diff: ({diff[h//2, w//2, 0]:.5f}, {diff[h//2, w//2, 1]:.5f}, {diff[h//2, w//2, 2]:.5f})")

# What does sRGB → linear give for the Non-Color value?
# Standard sRGB decoding: linear = ((srgb + 0.055) / 1.055)^2.4 for srgb > 0.04045
def srgb_to_linear(s):
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)

decoded = srgb_to_linear(arr)
print(f"\n=== applying sRGB→linear to Non-Color values ===")
print(f"  center decoded: ({decoded[h//2, w//2, 0]:.4f}, {decoded[h//2, w//2, 1]:.4f}, {decoded[h//2, w//2, 2]:.4f})")
print(f"  decoded mean: ({decoded[..., 0].mean():.4f}, {decoded[..., 1].mean():.4f}, {decoded[..., 2].mean():.4f})")
print(f"\n  → if the sRGB-tagged read matches this decoding, Blender DOES decode for sRGB.")
