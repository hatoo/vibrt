#include <optix.h>
#include "devicecode.h"

extern "C" {
__constant__ Params params;
}

// --- helpers ---

static __forceinline__ __device__ void setPayload(float r, float g, float b)
{
    optixSetPayload_0(__float_as_uint(r));
    optixSetPayload_1(__float_as_uint(g));
    optixSetPayload_2(__float_as_uint(b));
}

static __forceinline__ __device__ unsigned int packColor(float r, float g, float b)
{
    auto clamp01 = [](float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); };
    unsigned int ir = (unsigned int)(clamp01(r) * 255.0f);
    unsigned int ig = (unsigned int)(clamp01(g) * 255.0f);
    unsigned int ib = (unsigned int)(clamp01(b) * 255.0f);
    return (255u << 24) | (ib << 16) | (ig << 8) | ir; // ABGR
}

// --- programs ---

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const float u = (float(idx.x) + 0.5f) / float(dim.x);
    const float v = (float(idx.y) + 0.5f) / float(dim.y);

    const float2 d = make_float2(u * 2.0f - 1.0f, v * 2.0f - 1.0f);

    float3 origin = make_float3(params.cam_eye[0], params.cam_eye[1], params.cam_eye[2]);
    float3 U = make_float3(params.cam_u[0], params.cam_u[1], params.cam_u[2]);
    float3 V = make_float3(params.cam_v[0], params.cam_v[1], params.cam_v[2]);
    float3 W = make_float3(params.cam_w[0], params.cam_w[1], params.cam_w[2]);

    float3 direction;
    direction.x = d.x * U.x + d.y * V.x + W.x;
    direction.y = d.x * U.y + d.y * V.y + W.y;
    direction.z = d.x * U.z + d.y * V.z + W.z;
    float len = sqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    direction.x /= len;
    direction.y /= len;
    direction.z /= len;

    unsigned int p0, p1, p2;
    optixTrace(
        params.handle,
        origin,
        direction,
        0.0f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0, p1, p2);

    float r = __uint_as_float(p0);
    float g = __uint_as_float(p1);
    float b = __uint_as_float(p2);

    params.image[idx.y * params.image_width + idx.x] = packColor(r, g, b);
}

extern "C" __global__ void __miss__ms()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    setPayload(data->bg_color[0], data->bg_color[1], data->bg_color[2]);
}

extern "C" __global__ void __closesthit__ch()
{
    const float2 bary = optixGetTriangleBarycentrics();
    // Color from barycentric coordinates
    setPayload(bary.x, bary.y, 1.0f - bary.x - bary.y);
}
