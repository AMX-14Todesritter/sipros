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
             const mvh_cuda::Config& config,bool aggregate,int fusionMode) {
    if(aggregate && geometry!=GeometryKind::Spheres)throw std::invalid_argument("Aggregate replay requires spheres");
    if (activeGeometry && *activeGeometry != geometry)
        throw std::runtime_error("RT backend changed without reset");

    switch (geometry) {
    case GeometryKind::Spheres:
        sphere_backend::prepare(scans, devicePeaks, deviceClasses, peakCount, config,aggregate,fusionMode);
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
void launchMatching(const mvh_cuda::Scan* scans, const mvh_matching::Query* queries,
                    int count, int* results, const mvh_cuda::Config& config) {
    if (count < 0) throw std::invalid_argument("Negative matching query count");
    if (!count) return;
    stageMatching(scans, queries, count, results, config);
    launchStagedMatching();
}

void stageMatching(const mvh_cuda::Scan* scans, const mvh_matching::Query* queries,
                   int count, int* results, const mvh_cuda::Config& config) {
    if (count <= 0 || !scans || !queries || !results)
        throw std::invalid_argument("Invalid matching query buffers/count");
    if (!activeGeometry || *activeGeometry != GeometryKind::Spheres)
        throw std::runtime_error("Matching-only launch requires prepared sphere geometry");
    Params params{};
    params.scans = scans; params.matchQueries = queries;
    params.matchResults = results; params.size = count; params.cfg = config;
    sphere_backend::stage(params);
}

void launchStagedMatching() {
    if (!activeGeometry || *activeGeometry != GeometryKind::Spheres)
        throw std::runtime_error("Matching-only launch requires prepared sphere geometry");
    sphere_backend::launchStagedMatching();
}

void stageFusion(Params params) {
    if(!activeGeometry || *activeGeometry!=GeometryKind::Spheres || !params.fusionOutputs || !params.candidates || !params.fusionValidity)
        throw std::invalid_argument("Invalid fusion staging");
    sphere_backend::stage(params);
}

void stageGroupedMatching(const mvh_cuda::Scan* scans,
    const mvh_cuda::Candidate* groups, const uint64_t* offsets, const double* mz,
    uint64_t base, int count, int* results, const mvh_cuda::Config& config,
    mvh_matching::Statistics* statistics,mvh_matching::CandidateSummary* summaries,const int* classes) {
    if(count<=0 || !scans || !groups || !offsets || !mz || (!results && !summaries) || (summaries && !classes))
        throw std::invalid_argument("Invalid grouped matching buffers/count");
    if(!activeGeometry || *activeGeometry!=GeometryKind::Spheres)
        throw std::runtime_error("Grouped matching requires prepared spheres");
    Params params{};
    params.scans=scans;params.matchGroups=groups;params.matchOffsets=offsets;
    params.matchMz=mz;params.matchBase=base;params.size=count;
    params.matchResults=results;params.cfg=config;
    params.matchStatistics=statistics;params.matchSummaries=summaries;params.classes=classes;
    sphere_backend::stage(params);
}

}
