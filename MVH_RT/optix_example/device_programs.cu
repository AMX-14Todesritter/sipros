#include "scene.h"
#include <optix_device.h>
#include <math_constants.h>

// The name must match pipelineLaunchParamsVariableName on the host.
extern "C" {
__constant__ LaunchParams params;
}

extern "C" __global__ void __raygen__camera() {
    const uint3 pixel = optixGetLaunchIndex();
    const unsigned int index = pixel.y * params.width + pixel.x;
    const Ray& ray = params.rays[index];

    // Each trace has its own result payload.
    unsigned int rightId = 0xffffffffu;
    unsigned int rightTBits = __float_as_uint(CUDART_INF_F);

    unsigned int leftId = 0xffffffffu;
    unsigned int leftTBits = __float_as_uint(CUDART_INF_F);

    // Search along +x.
    optixTrace(
        params.traversable,
        ray.origin,
        make_float3(1.0f, 0.0f, 0.0f),
        ray.tmin, ray.tmax, 0.0f,
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        rightId, rightTBits
    );

    const bool rightHit = rightId != 0xffffffffu;
    const float rightT = __uint_as_float(rightTBits);

    // Use the first hit as the upper bound for the opposite direction.
    const float leftTmax =
        rightHit ? fminf(ray.tmax, rightT) : ray.tmax;

    // Search along -x from the same origin.
    optixTrace(
        params.traversable,
        ray.origin,
        make_float3(-1.0f, 0.0f, 0.0f),
        ray.tmin, leftTmax, 0.0f,
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        leftId, leftTBits
    );

    // Keep the first trace result by default.
    unsigned int bestId = rightId;
    float bestT = rightT;

    // Replace it only if the second trace found a better result.
    const bool leftHit = leftId != 0xffffffffu;
    const float leftT = __uint_as_float(leftTBits);

    if (leftHit && (!rightHit || leftT < rightT)) {
        bestId = leftId;
        bestT = leftT;
    }

    // Store one final result for this origin.
    params.results[index] = {
        bestId == 0xffffffffu ? -1 : static_cast<int>(bestId),
        bestT
    };
}


extern "C" __global__ void __closesthit__record() {
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __miss__background() {
    optixSetPayload_0(0xffffffffu);
    optixSetPayload_1(__float_as_uint(CUDART_INF_F));
}
