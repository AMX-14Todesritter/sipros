cmake_minimum_required(VERSION 3.18)
project(OptixSphereTutorial LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
set(OPTIX_ROOT "" CACHE PATH "Existing NVIDIA OptiX 9 SDK or optix-dev checkout")
find_path(OPTIX_INCLUDE_DIR optix.h HINTS "${OPTIX_ROOT}/include" "${OPTIX_ROOT}")
if(NOT OPTIX_INCLUDE_DIR)
  message(FATAL_ERROR "OptiX headers not found. Supply -DOPTIX_ROOT=/path/to/existing/SDK. No dependency is downloaded automatically.")
endif()
file(STRINGS "${OPTIX_INCLUDE_DIR}/optix.h" OPTIX_VERSION_LINE REGEX "^#define OPTIX_VERSION ")
string(REGEX MATCH "[0-9]+" OPTIX_HEADER_VERSION "${OPTIX_VERSION_LINE}")
if(NOT OPTIX_HEADER_VERSION OR OPTIX_HEADER_VERSION LESS 90000)
  message(FATAL_ERROR "This example targets OptiX 9.x, including Blackwell support.")
endif()
set(DEVICE_PTX "${CMAKE_CURRENT_BINARY_DIR}/device_programs.ptx")
add_custom_command(OUTPUT "${DEVICE_PTX}"
  COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --std=c++17 --gpu-architecture=compute_70
          --generate-line-info --fmad=false "-I${OPTIX_INCLUDE_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/device_programs.cu" -o "${DEVICE_PTX}"
  DEPENDS device_programs.cu scene.h VERBATIM)
add_custom_target(optix_device_programs DEPENDS "${DEVICE_PTX}")
add_library(mvh_rt_support STATIC rt_support.cpp)

target_include_directories(mvh_rt_support PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${OPTIX_INCLUDE_DIR}"
)

target_link_libraries(mvh_rt_support PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)

add_executable(optix_spheres main.cpp)
target_link_libraries(optix_spheres PRIVATE mvh_rt_support)
add_dependencies(optix_spheres optix_device_programs)
set_target_properties(optix_spheres PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
