#pragma once
#include <cuda_runtime.h>
#include <optix.h>

// These POD structures are shared by the host and device programs.
// An ExperimentPeak is our primitive; its AABB is only a conservative traversal bound.
struct ScanPeak {
    float3 mz;
    int scanId;
    int intensityClass;
};

struct Ray {
    float3 origin;
    double mz;
    float tmin;
    float tmax;
};

struct RayResult {
    int primitiveId;  // -1 means miss; otherwise an index into the peak array.
    float distance;   // Ray parameter t; directions in this example have unit length.
};

struct LaunchParams {
    OptixTraversableHandle traversable;
    const ScanPeak *peaks;
    const Ray *rays;
    RayResult *results;
    uchar4 *pixels;
    unsigned int width;
    unsigned int height;
};

// One hitgroup SBT record serves all custom primitives in this single GAS.
struct HitGroupData {
    const ScanPeak *peaks;
};

__host__ __device__ inline float3 subtract(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float3 rayOrigin(unsigned int x, unsigned int y,
                                          unsigned int width, unsigned int height) {
    // Orthographic camera at z=-3. Each pixel has a different origin and the
    // same +z direction, making the geometry easy to inspect and modify.
    const float aspect = float(width) / float(height);
    return make_float3(((float(x) + 0.5f) / width * 2.0f - 1.0f) * aspect * 1.2f,
                       (1.0f - (float(y) + 0.5f) / height * 2.0f) * 1.2f, -3.0f);
}
