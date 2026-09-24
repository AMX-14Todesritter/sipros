#include "bridge.h"
#include "backends.h"
#include <optional>

namespace mvh_rt_gpu {
namespace {
std::optional<GeometryKind> activeGeometry;
}

void reset() {
    sphere_backend::reset();
    triangle_backend::reset();
    activeGeometry.reset();
}

void prepare(const std::vector<mvh_cuda::Scan>& scans,
             const mvh_cuda::Scan* deviceScans, const double* devicePeaks,
             const int* deviceClasses, size_t peakCount, GeometryKind geometry,
             const mvh_cuda::Config& config) {
    if (activeGeometry && *activeGeometry != geometry)
        throw std::runtime_error("RT backend changed without reset");

    switch (geometry) {
    case GeometryKind::Spheres:
        sphere_backend::prepare(scans, devicePeaks, deviceClasses, peakCount, config);
        break;
    case GeometryKind::Triangles:
    case GeometryKind::InstancedTriangles:
        triangle_backend::prepare(scans, deviceScans, devicePeaks, peakCount,
                                  geometry == GeometryKind::InstancedTriangles);
        break;
    default:
        throw std::runtime_error("Unknown RT geometry kind");
    }
    // A failed setup must not mark the backend as ready for launch.
    activeGeometry = geometry;
}

void launch(Params params) {
    if (!params.size) return;
    if (!activeGeometry) throw std::runtime_error("RT resources not prepared");
    if (*activeGeometry == GeometryKind::Spheres)
        sphere_backend::launch(params);
    else
        triangle_backend::launch(params);
}
}
