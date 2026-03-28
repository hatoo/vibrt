#pragma once

#include <optix_types.h>

struct Params
{
    unsigned int* image;
    unsigned int  image_width;
    unsigned int  image_height;
    float         cam_eye[3];
    float         cam_u[3];
    float         cam_v[3];
    float         cam_w[3];
    OptixTraversableHandle handle;
};

struct RayGenData
{
    // empty
};

struct MissData
{
    float bg_color[3];
};

struct HitGroupData
{
    // empty
};
