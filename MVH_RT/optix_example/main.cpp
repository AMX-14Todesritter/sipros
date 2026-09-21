// Standalone OptiX tutorial: two custom spheres, one GAS, one ray per pixel.
// This file deliberately does not include or link any Sipros/MVH code.
#include "rt_support.h"
#include <optix_stubs.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <sstream>

namespace fs = std::filesystem;


//read test data from file
std::map<double, char> loadTestPeakData(const fs::path& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Cannot open: " + path.string());

    std::string line;
    std::getline(input, line);

    if (!line.empty() && line.back() == '\r')
        line.pop_back();

    if (line != "peak_index\tmz\tintensity_class")
        throw std::runtime_error("Unexpected peak file header");

    std::map<double, char> peakData;
    size_t expectedIndex = 0;
    double previousMz = -std::numeric_limits<double>::infinity();

    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos)
            continue;

        std::istringstream row(line);
        size_t index;
        double mz;
        int intensityClass;
        std::string extra;

        if (!(row >> index >> mz >> intensityClass) || (row >> extra))
            throw std::runtime_error("Invalid peak row: " + line);

        // The exported snapshot has consecutive indices and increasing m/z.
        if (index != expectedIndex ||
            !std::isfinite(mz) || mz <= previousMz ||
            intensityClass < 1 || intensityClass > 3)
            throw std::runtime_error("Invalid peak data: " + line);

        peakData.emplace(mz, static_cast<char>(intensityClass));

        previousMz = mz;
        ++expectedIndex;
    }

    if (input.bad())
        throw std::runtime_error("Failed to read peak file");

    if (peakData.empty())
        throw std::runtime_error("Peak file is empty");

    return peakData;
}

std::vector<double> loadTestIonData(const fs::path& path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Cannot open: " + path.string());

    std::string line;
    std::getline(input, line);

    if (!line.empty() && line.back() == '\r')
        line.pop_back();

    if (line !=
        "ion_index\tion_label\tfragment_charge\tmz\tin_global_mz_range")
        throw std::runtime_error("Unexpected theoretical ion header");

    std::vector<double> sequenceIonMasses;

    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos)
            continue;

        std::istringstream row(line);

        size_t ionIndex;
        std::string ionLabel;
        int fragmentCharge;
        double mz;
        int inGlobalMzRange;
        std::string extra;

        if (!(row >> ionIndex >> ionLabel >> fragmentCharge
                  >> mz >> inGlobalMzRange) ||
            (row >> extra))
            throw std::runtime_error("Invalid ion row: " + line);

        if (ionIndex != sequenceIonMasses.size() ||
            !std::isfinite(mz) || mz <= 0.0 ||
            fragmentCharge <= 0 ||
            (inGlobalMzRange != 0 && inGlobalMzRange != 1))
            throw std::runtime_error("Invalid ion data: " + line);

        // Preserve the original ion order, including repeated m/z values.
        sequenceIonMasses.push_back(mz);
    }

    if (input.bad())
        throw std::runtime_error("Failed to read theoretical ion file");

    if (sequenceIonMasses.empty())
        throw std::runtime_error("Theoretical ion file is empty");

    return sequenceIonMasses;
}


void writeResults(
    const fs::path& output,
    const std::vector<Ray>& rays,
    const std::vector<ScanPeak>& peaks,
    const std::vector<RayResult>& results)
{
    fs::create_directories(output);

    std::ofstream table;
    table.exceptions(std::ios::failbit | std::ios::badbit);
    table.open(output / "matches.tsv");

    table << std::setprecision(17);
    table << "ray_index\ttheoretical_mz\tpeak_index\t"
             "experimental_mz\tintensity_class\tt\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];

        table << i << '\t'
              << rays.at(i).mz << '\t'
              << result.primitiveId;

        if (result.primitiveId >= 0) {
            const auto& peak =
                peaks.at(static_cast<size_t>(result.primitiveId));

            table << '\t' << peak.mz.x
                  << '\t' << peak.intensityClass
                  << '\t' << result.distance;
        } else {
            table << "\t\t\t";
        }

        table << '\n';
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: optix_spheres device_programs.ptx NEW_OUTPUT_DIRECTORY");
        const fs::path output = fs::absolute(argv[2]);
        if (fs::exists(output)) throw std::runtime_error("Output directory already exists");
        OptixObjects objects;
        initializeOptix(objects);

        // Load classified experimental peaks from the saved scan.
        const auto peakData = loadTestPeakData(
            "MVH_RT/test_data/experimental_peaks.tsv"
        );

        // Adapt the test snapshot to the original PeakList array layout.
        std::vector<double> observedMz;
        std::vector<char> observedClasses;

        observedMz.reserve(peakData.size());
        observedClasses.reserve(peakData.size());

        for (const auto& entry : peakData) {
            observedMz.push_back(entry.first);
            observedClasses.push_back(entry.second);
        }

        ScanRtResources scanResources(
            objects.context,
            observedMz,
            observedClasses,
            1004
        );

        createPipeline(objects, argv[1]);

        // Example input; later replace this with the generated ion list.
        const std::vector<double> sequenceIonMasses = loadTestIonData(
            "MVH_RT/test_data/theoretical_ions.tsv"
        );

        const std::vector<Ray> rays = generateRays(
            sequenceIonMasses.data(),
            sequenceIonMasses.size(),
            0.01
        );

        if (rays.empty())
            throw std::runtime_error("Ray list is empty");

        // One launch entry per ray.
        const auto hostResults = traceRays(
            objects,
            scanResources,
            rays
        );

        const auto repeatedResults = traceRays(
            objects,
            scanResources,
            rays
        );

        if (hostResults.size() != repeatedResults.size())
            throw std::runtime_error("Repeated query size mismatch");

        for (size_t i = 0; i < hostResults.size(); ++i) {
            if (hostResults[i].primitiveId != repeatedResults[i].primitiveId ||
                hostResults[i].distance != repeatedResults[i].distance)
                throw std::runtime_error("Repeated query result mismatch");
        }

        std::cout << "PASS: repeated query reused the same GAS\n";

        writeResults(output, rays, scanResources.peaks(), hostResults);

        std::cout << "Wrote " << output / "matches.tsv" << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
