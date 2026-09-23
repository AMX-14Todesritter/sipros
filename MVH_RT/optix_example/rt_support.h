#pragma once

#include "scene.h"
#include <filesystem>
#include <vector>
#include <memory>

void checkCuda(cudaError_t result);
void checkOptix(OptixResult result);

// RAII keeps device buffers alive until traversal finishes and frees them on errors.
template <typename T> class DeviceBuffer {
public:
    T *data = nullptr;
    size_t count = 0;
    explicit DeviceBuffer(size_t size) : count(size) {
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));
    }
    explicit DeviceBuffer(const std::vector<T> &values) : DeviceBuffer(values.size()) {
        if (count) checkCuda(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    CUdeviceptr address() const { return reinterpret_cast<CUdeviceptr>(data); }
    std::vector<T> download() const {
        std::vector<T> values(count);
        if (count) checkCuda(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

struct OptixObjects {
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup raygen = nullptr, miss = nullptr, hit = nullptr;
    OptixPipeline pipeline = nullptr;

    OptixObjects() = default;
    ~OptixObjects();

    OptixObjects(const OptixObjects&) = delete;
    OptixObjects& operator=(const OptixObjects&) = delete;
};

template <typename Data> struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    Data data;
};
struct EmptyData { int unused = 0; };

void initializeOptix(OptixObjects& objects);

void createPipeline(
    OptixObjects& objects,
    const std::filesystem::path& ptxPath, bool instanced = false);

std::vector<ScanPeak> makeScene(
    const std::vector<double>& mzValues,
    const std::vector<char>& classes,
    int scanId);

std::vector<float3> makeVertices(
    const std::vector<ScanPeak>& peaks);

Ray makeRay(double mz, double tolerance);

std::vector<Ray> generateRays(
    const double* mzValues,
    size_t count,
    double tolerance);

//BVH Memory share
class ScanRtResources {
public:
    ScanRtResources(
        OptixDeviceContext context,
        const std::vector<double>& mzValues,
        const std::vector<char>& classes,
        int scanId);

    ScanRtResources(const ScanRtResources&) = delete;
    ScanRtResources& operator=(const ScanRtResources&) = delete;

    OptixTraversableHandle handle() const {
        return gas_;
    }

    const ScanPeak* devicePeaks() const {
        return devicePeaks_.data;
    }

    const std::vector<ScanPeak>& peaks() const {
        return peaks_;
    }

private:
    std::vector<ScanPeak> peaks_;
    DeviceBuffer<ScanPeak> devicePeaks_;
    DeviceBuffer<float3> deviceVertices_;
    std::unique_ptr<DeviceBuffer<unsigned char>> gasMemory_;
    OptixTraversableHandle gas_ = 0;
};

//query RT
std::vector<RayResult> traceRays(
    const OptixObjects& objects,
    const ScanRtResources& scan,
    const std::vector<Ray>& rays);