// Packed image storage adapter for Cycles' unmodified CPU interpolation
// kernels.
using namespace vibrt_cycles;
using float4 = vibrt_cycles::float4;
using vibrt_cycles::make_float4;
#define ccl_always_inline VN_INLINE
enum { EXTENSION_REPEAT, EXTENSION_EXTEND, EXTENSION_CLIP, EXTENSION_MIRROR };
enum { INTERPOLATION_LINEAR, INTERPOLATION_CLOSEST };
struct ImageView {
  const float *pixels;
  int channels;
  bool associate_alpha;
  bool equirectangular;
};
struct KernelImageInfo {
  int width, height, extension, interpolation;
  const ImageView *data;
};
struct ImageReader {
  using TexT = ImageView;
  using OutT = float4;
  static VN_INLINE OutT zero() { return make_float4(0); }
  static VN_INLINE OutT read(const TexT *view, int x, int y, int width, int) {
    const float *p = view->pixels + (y * width + x) * view->channels;
    float alpha = view->channels == 4 ? p[3] : (view->channels == 2 ? p[1] : 1);
    float a = view->associate_alpha ? alpha : 1;
    return make_float4(p[0] * a, (view->channels <= 2 ? p[0] : p[1]) * a,
                       (view->channels <= 2 ? p[0] : p[2]) * a, alpha);
  }
  static VN_INLINE OutT read_clip(const TexT *view, int x, int y, int width,
                                  int height) {
    if (view->equirectangular) {
      // Longitude repeats; each latitude neighbor is clamped separately at the
      // poles.
      x = ((x % width) + width) % width;
      y = clamp(y, 0, height - 1);
    } else if (x < 0 || x >= width || y < 0 || y >= height)
      return zero();
    return read(view, x, y, width, height);
  }
};
