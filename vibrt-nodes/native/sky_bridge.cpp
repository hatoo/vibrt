// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-FileCopyrightText: 2026 hatoo
// SPDX-License-Identifier: Apache-2.0
//
// Analytic precomputation/evaluation adapted from Blender v5.1.1:
// intern/cycles/scene/shader_nodes.cpp and intern/cycles/kernel/svm/sky.h.
// Standalone ownership, image generation and filtering supplied by vibrt.
#include "sky_bridge.h"
#include "sky_hosek.h"
#include "sky_nishita.h"
#include "sky_math.h"

static float2 sky_spherical_coordinates(const float3 dir)
{
  return make_float2(acosf(dir.z), atan2f(dir.x, dir.y));
}

struct SunSky {
  /* sun direction in spherical and cartesian */
  float theta, phi;

  /* Parameter */
  float radiance_x, radiance_y, radiance_z;
  float config_x[9], config_y[9], config_z[9], nishita_data[11];
};

/* Preetham model */
static float sky_perez_function(const float lam[6], float theta, const float gamma)
{
  return (1.0f + lam[0] * expf(lam[1] / cosf(theta))) *
         (1.0f + lam[2] * expf(lam[3] * gamma) + lam[4] * cosf(gamma) * cosf(gamma));
}

static void sky_texture_precompute_preetham(SunSky *sunsky,
                                            const float3 dir,
                                            const float turbidity)
{
  /*
   * We re-use the SunSky struct of the new model, to avoid extra variables
   * zenith_Y/x/y is now radiance_x/y/z
   * perez_Y/x/y is now config_x/y/z
   */

  const float2 spherical = sky_spherical_coordinates(dir);
  const float theta = spherical.x;
  const float phi = spherical.y;

  sunsky->theta = theta;
  sunsky->phi = phi;

  const float theta2 = theta * theta;
  const float theta3 = theta2 * theta;
  const float T = turbidity;
  const float T2 = T * T;

  const float chi = (4.0f / 9.0f - T / 120.0f) * (M_PI_F - 2.0f * theta);
  sunsky->radiance_x = (4.0453f * T - 4.9710f) * tanf(chi) - 0.2155f * T + 2.4192f;
  sunsky->radiance_x *= 0.06f;

  sunsky->radiance_y = (0.00166f * theta3 - 0.00375f * theta2 + 0.00209f * theta) * T2 +
                       (-0.02903f * theta3 + 0.06377f * theta2 - 0.03202f * theta + 0.00394f) * T +
                       (0.11693f * theta3 - 0.21196f * theta2 + 0.06052f * theta + 0.25886f);

  sunsky->radiance_z = (0.00275f * theta3 - 0.00610f * theta2 + 0.00317f * theta) * T2 +
                       (-0.04214f * theta3 + 0.08970f * theta2 - 0.04153f * theta + 0.00516f) * T +
                       (0.15346f * theta3 - 0.26756f * theta2 + 0.06670f * theta + 0.26688f);

  sunsky->config_x[0] = (0.1787f * T - 1.4630f);
  sunsky->config_x[1] = (-0.3554f * T + 0.4275f);
  sunsky->config_x[2] = (-0.0227f * T + 5.3251f);
  sunsky->config_x[3] = (0.1206f * T - 2.5771f);
  sunsky->config_x[4] = (-0.0670f * T + 0.3703f);

  sunsky->config_y[0] = (-0.0193f * T - 0.2592f);
  sunsky->config_y[1] = (-0.0665f * T + 0.0008f);
  sunsky->config_y[2] = (-0.0004f * T + 0.2125f);
  sunsky->config_y[3] = (-0.0641f * T - 0.8989f);
  sunsky->config_y[4] = (-0.0033f * T + 0.0452f);

  sunsky->config_z[0] = (-0.0167f * T - 0.2608f);
  sunsky->config_z[1] = (-0.0950f * T + 0.0092f);
  sunsky->config_z[2] = (-0.0079f * T + 0.2102f);
  sunsky->config_z[3] = (-0.0441f * T - 1.6537f);
  sunsky->config_z[4] = (-0.0109f * T + 0.0529f);

  /* unused for old sky model */
  for (int i = 5; i < 9; i++) {
    sunsky->config_x[i] = 0.0f;
    sunsky->config_y[i] = 0.0f;
    sunsky->config_z[i] = 0.0f;
  }

  sunsky->radiance_x /= sky_perez_function(sunsky->config_x, 0, theta);
  sunsky->radiance_y /= sky_perez_function(sunsky->config_y, 0, theta);
  sunsky->radiance_z /= sky_perez_function(sunsky->config_z, 0, theta);
}

/* Hosek / Wilkie */
static void sky_texture_precompute_hosek(SunSky *sunsky,
                                         const float3 dir,
                                         float turbidity,
                                         const float ground_albedo)
{
  /* Calculate Sun Direction and save coordinates */
  const float2 spherical = sky_spherical_coordinates(dir);
  float theta = spherical.x;
  const float phi = spherical.y;

  /* Clamp Turbidity */
  turbidity = clamp(turbidity, 0.0f, 10.0f);

  /* Clamp to Horizon */
  theta = clamp(theta, 0.0f, M_PI_2_F);

  sunsky->theta = theta;
  sunsky->phi = phi;

  const float solarElevation = M_PI_2_F - theta;

  /* Initialize Sky Model */
  SKY_ArHosekSkyModelState *sky_state;
  sky_state = SKY_arhosek_xyz_skymodelstate_alloc_init(
      (double)turbidity, (double)ground_albedo, (double)solarElevation);

  /* Copy values from sky_state to SunSky */
  for (int i = 0; i < 9; ++i) {
    sunsky->config_x[i] = (float)sky_state->configs[0][i];
    sunsky->config_y[i] = (float)sky_state->configs[1][i];
    sunsky->config_z[i] = (float)sky_state->configs[2][i];
  }
  sunsky->radiance_x = (float)sky_state->radiances[0];
  sunsky->radiance_y = (float)sky_state->radiances[1];
  sunsky->radiance_z = (float)sky_state->radiances[2];

  /* Free sky_state */
  SKY_arhosekskymodelstate_free(sky_state);
}

/* Nishita improved */
struct Sky {
  VibrtSkySettings settings;
  SunSky analytic{};
  std::vector<float> atmosphere;
  float3 sun_bottom{0.0f}, sun_top{0.0f}, sun_dir{0.0f};
  float earth_angle = 0;

  explicit Sky(const VibrtSkySettings &p) : settings(p) {
    const float3 direction = normalize(make_float3(p.sun_direction[0], p.sun_direction[1], p.sun_direction[2]));
    if (p.model == 0) sky_texture_precompute_preetham(&analytic, direction, p.turbidity);
    else if (p.model == 1) sky_texture_precompute_hosek(&analytic, direction, p.turbidity, p.ground_albedo);
    else {
      atmosphere.resize(512*256*3);
      auto texture = p.model == 2 ? SKY_single_scattering_precompute_texture : SKY_multiple_scattering_precompute_texture;
      texture(atmosphere.data(), 3, 512, 256, p.sun_elevation, p.altitude,
              p.air_density, p.aerosol_density, p.ozone_density);
      sun_dir = make_float3(-cosf(p.sun_elevation)*sinf(p.sun_rotation),
                            cosf(p.sun_elevation)*cosf(p.sun_rotation), sinf(p.sun_elevation));
      earth_angle = -SKY_earth_intersection_angle(p.altitude);
      if (p.sun_disc && p.sun_size > 0.0f) {
        float bottom[3], top[3];
        if (p.model == 2)
          SKY_single_scattering_precompute_sun(p.sun_elevation, p.sun_size, p.altitude,
                                              p.air_density, p.aerosol_density, bottom, top);
        else
          SKY_multiple_scattering_precompute_sun(p.sun_elevation, p.sun_size, p.altitude,
                                                 p.air_density, p.aerosol_density, p.ozone_density, bottom, top);
        sun_bottom = make_float3(bottom[0], bottom[1], bottom[2]);
        sun_top = make_float3(top[0], top[1], top[2]);
      }
    }
  }

  float3 to_rgb(float3 xyz) const {
    float3 rgb;
    for (int i = 0; i < 3; ++i)
      rgb[i] = std::max(0.0f, settings.xyz_to_rgb[i*3]*xyz.x +
                             settings.xyz_to_rgb[i*3+1]*xyz.y + settings.xyz_to_rgb[i*3+2]*xyz.z);
    return rgb;
  }

  float3 lookup(float u, float v) const {
    // The shader wraps longitude; the atmosphere image uses EXTENSION_EXTEND.
    const float x = (u-floorf(u))*512.0f - 0.5f;
    const float y = clamp(v,0.0f,1.0f)*256.0f - 0.5f;
    const int x0 = int(floorf(x)), y0 = int(floorf(y));
    const float fx = x-x0, fy = y-y0;
    auto pixel = [&](int ix, int iy) {
      const size_t offset = (std::clamp(iy,0,255)*512 + std::clamp(ix,0,511))*3;
      return make_float3(atmosphere[offset], atmosphere[offset+1], atmosphere[offset+2]);
    };
    return mix(mix(pixel(x0,y0),pixel(x0+1,y0),fx),
               mix(pixel(x0,y0+1),pixel(x0+1,y0+1),fx),fy);
  }

  float3 evaluate(float3 dir) const {
    const float theta = acosf(clamp(dir.z,-1.0f,1.0f));
    const float phi = atan2f(dir.y,dir.x);
    if (settings.model < 2) {
      const float gamma = acosf(clamp(sinf(theta)*sinf(analytic.theta)*
          cosf((M_PI_2_F-phi)-analytic.phi) + cosf(theta)*cosf(analytic.theta),-1.0f,1.0f));
      const float t = std::min(theta,M_PI_2_F-0.001f);
      const float radiance[3] = {analytic.radiance_x,analytic.radiance_y,analytic.radiance_z};
      const float *configs[3] = {analytic.config_x,analytic.config_y,analytic.config_z};
      float3 value;
      for (int i=0; i<3; ++i) {
        const float *c = configs[i];
        const float ct = cosf(t), cg = cosf(gamma);
        if (settings.model == 0) value[i] = radiance[i]*sky_perez_function(c,t,gamma);
        else {
          const float mie = (1+cg*cg)/powf(1+c[8]*c[8]-2*c[8]*cg,1.5f);
          value[i] = radiance[i]*(1+c[0]*expf(c[1]/(ct+0.01f)))*
              (c[2]+c[3]*expf(c[4]*gamma)+c[5]*cg*cg+c[6]*mie+c[7]*sqrtf(ct));
        }
      }
      if (settings.model == 0) {
        value = value.z == 0 ? float3(0.0f) :
            make_float3(value.y*value.x/value.z,value.x,(1-value.y-value.z)*value.x/value.z);
        return to_rgb(value);
      }
      return to_rgb(value)*(M_2PI_F/683.0f);
    }
    const float elevation = M_PI_2_F-theta;
    const float u = (-phi-M_PI_2_F+settings.sun_rotation)/M_2PI_F;
    const float v = copysignf(sqrtf(fabsf(elevation)/M_PI_2_F),elevation)*0.5f+0.5f;
    float3 xyz = lookup(u,v);
    if (settings.sun_disc && settings.sun_size > 0 && elevation > earth_angle) {
      // Chord length is well-conditioned for sub-degree sun discs.
      const float angle = 2*asinf(std::min(1.0f,len(dir-sun_dir)*0.5f));
      const float half = settings.sun_size*0.5f;
      if (angle < half) {
        const float vertical = (elevation-settings.sun_elevation)/settings.sun_size+0.5f;
        const float limb = 1-0.6f*(1-safe_sqrtf(1-sqr(angle/half)));
        xyz += mix(sun_bottom,sun_top,vertical)*settings.sun_intensity*limb;
      }
    }
    return to_rgb(xyz);
  }
};

extern "C" void *vibrt_sky_create(const VibrtSkySettings *settings) {
  try { return new Sky(*settings); } catch (...) { return nullptr; }
}
extern "C" void vibrt_sky_free(void *sky) { delete static_cast<Sky *>(sky); }
extern "C" void vibrt_sky_row(const void *opaque, int width, int height, int y, float *rgb) {
  const auto &sky = *static_cast<const Sky *>(opaque);
  auto direction = [&](float x, float row) {
    const float theta = M_PI_F*row/height, phi = M_2PI_F*x/width;
    const float3 dir = make_float3(sinf(theta)*cosf(phi),sinf(theta)*sinf(phi),cosf(theta));
    return sky.settings.invert_direction ? -dir : dir;
  };
  for (int x=0; x<width; ++x) {
    int samples = 2;
    if (sky.settings.model >= 2 && sky.settings.sun_disc && sky.settings.sun_size > 0) {
      const float angle = 2*asinf(std::min(1.0f,len(direction(x+0.5f,y+0.5f)-sky.sun_dir)*0.5f));
      if (angle < sky.settings.sun_size*0.5f+M_2PI_F/width+M_PI_F/height) samples = 16;
    }
    float3 sum(0.0f);
    for (int sy=0; sy<samples; ++sy)
      for (int sx=0; sx<samples; ++sx)
        sum += sky.evaluate(direction(x+(sx+0.5f)/samples,y+(sy+0.5f)/samples));
    for (int c=0; c<3; ++c) rgb[x*3+c] = sum[c]/float(samples*samples);
  }
}
