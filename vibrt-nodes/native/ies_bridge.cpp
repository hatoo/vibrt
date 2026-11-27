// Cycles owns parsing, photometric conversion, symmetry and packing.
#include "cycles_ies_host.h"
using namespace vibrt_ies_host;

extern "C" size_t vibrt_ies_parse(const char *text, size_t length, float *output, size_t capacity) {
  try {
    IESFile file;
    if (!file.load(string(text, length)))
      return 0;
    size_t size = file.packed_size();
    if (output && size <= capacity)
      file.pack(output);
    return size;
  } catch (...) {
    return 0;
  }
}

class Profile : public IESFile {
public:
  bool assign(const float *v, size_t nv, const float *h, size_t nh, const float *values) {
    type = TYPE_C;
    v_angles.assign(v, v + nv);
    h_angles.assign(h, h + nh);
    intensity.resize(nh);
    for (size_t i = 0; i < nh; ++i)
      intensity[i].assign(values + i * nv, values + (i + 1) * nv);
    return process();
  }
};
extern "C" size_t vibrt_ies_pack(const float *v, size_t nv, const float *h, size_t nh,
                                 const float *values, float *output, size_t capacity) {
  try {
    Profile file;
    if (!file.assign(v, nv, h, nh, values))
      return 0;
    size_t size = file.packed_size();
    if (output && size <= capacity)
      file.pack(output);
    return size;
  } catch (...) {
    return 0;
  }
}
