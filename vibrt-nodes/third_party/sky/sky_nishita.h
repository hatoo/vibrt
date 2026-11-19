// vibrt declarations for the standalone atmosphere implementations.
// SPDX-License-Identifier: MIT
#pragma once

void SKY_single_scattering_precompute_texture(float *, int, int, int,
                                               float, float, float, float, float);
void SKY_multiple_scattering_precompute_texture(float *, int, int, int,
                                                 float, float, float, float, float);
void SKY_single_scattering_precompute_sun(float, float, float, float, float, float *, float *);
void SKY_multiple_scattering_precompute_sun(float, float, float, float, float, float, float *, float *);
float SKY_earth_intersection_angle(float);
