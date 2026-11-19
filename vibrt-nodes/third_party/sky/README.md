# Native sky models

The original model files are copied unchanged from [Blender v5.1.1](https://github.com/blender/blender/tree/v5.1.1/intern/sky):

- `sky_hosek.cpp`, `sky_hosek_data.h`, `sky_hosek.h`: BSD-3-Clause.
- `sky_single_scattering.cpp`: Apache-2.0.
- `sky_multiple_scattering.cpp`: MIT.

`sky_math.h` and `sky_nishita.h` are vibrt's own standalone support headers.
`../../native/sky_bridge.cpp` adapts the Cycles analytic sky evaluator; Rust owns
the settings and output buffers. Cargo compiles these sources with `cc` and
requires a C++17 compiler. There is no Blender binary or TBB dependency.

Copyright notices and full license texts ship in
[`blender/vibrt_blender/licenses`](../../../blender/vibrt_blender/licenses/NOTICE.txt).

The addon supports Blender 5.1's default Linear Rec.709 working space. Direct
sky nodes, RGB Curves (including HDR extrapolation), Hue/Saturation, Gamma,
Invert, supported color mixes and background shader mixes run in-process.
Unsupported coordinate/node graphs, custom OCIO and other Blender versions
keep the isolated Cycles worker. `VIBRT_CYCLES_SKY=1` forces that reference path.

Run `blender -b --factory-startup --python-exit-code 1 --python scripts/test_native_sky.py`
from the repository root to compare the models and graphs against Cycles.
