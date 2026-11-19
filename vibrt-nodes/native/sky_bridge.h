// SPDX-License-Identifier: MIT
#pragma once

// All fields are 32-bit and in the same order as NativeSky in sky.rs.
struct VibrtSkySettings {
  int model;
  float sun_direction[3];
  float turbidity, ground_albedo;
  float sun_elevation, sun_rotation, sun_size, sun_intensity;
  float altitude, air_density, aerosol_density, ozone_density;
  int sun_disc;
  int invert_direction;
  float xyz_to_rgb[9];
};
static_assert(sizeof(VibrtSkySettings) == 100, "Rust/C++ sky settings layout");
extern "C" {
void *vibrt_sky_create(const VibrtSkySettings *settings);
void vibrt_sky_free(void *sky);
void vibrt_sky_row(const void *sky, int width, int height, int y, float *rgb);
}
