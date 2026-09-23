#pragma once

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

namespace sipros_output {

inline std::filesystem::path root() {
    const char *configured = std::getenv("SIPROS_OUTPUT_ROOT");
    const std::filesystem::path directory =
        configured && *configured ? configured : "output";
    return (std::filesystem::path(SIPROS_PROJECT_ROOT) / directory).lexically_normal();
}

inline std::filesystem::path searchDirectory(const std::string &component,
                                             const std::string &input) {
    std::string label = std::filesystem::path(input).stem().string();
    for (char &character : label) {
        const bool safe = (character >= 'a' && character <= 'z') ||
                          (character >= 'A' && character <= 'Z') ||
                          (character >= '0' && character <= '9') ||
                          character == '_' || character == '-';
        if (!safe) character = '_';
    }
    if (label.empty()) label = "run";
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now - seconds).count();
    const std::time_t time = std::chrono::system_clock::to_time_t(seconds);
    std::ostringstream name;
    // CLI path selection runs on the main thread before worker threads start.
    name << label << '_' << std::put_time(std::gmtime(&time), "%Y%m%dT%H%M%S")
         << '_' << std::setfill('0') << std::setw(6) << micros << 'Z';
    return root() / "search" / component / name.str();
}

} // namespace sipros_output
