#include "devicecode.h"
#include <optix.h>
#include "rect_light_sampling.h"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef INV_PIf
#define INV_PIf 0.31830988618379067154f
#endif

extern "C" {
__constant__ LaunchParams params;
}

#include "device/math.h"

#include "device/volume.h"

#include "device/surface.h"

#include "device/environment.h"

#include "device/visibility.h"

#include "device/material.h"

#include "device/lighting.h"

#include "device/geometry.h"
#include "device/scene_query.h"

#include "device/programs.h"

#include "device/integrator.h"
