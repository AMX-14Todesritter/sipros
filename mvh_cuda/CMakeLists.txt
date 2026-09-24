cmake_minimum_required(VERSION 3.18)
project(SiprosMvhCuda LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenMP REQUIRED)
get_filename_component(SIPROS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
# Shared vendor dependency; no duplicated MSToolkit checkout.
include_directories("${SIPROS_ROOT}/MSToolkit/include")
add_subdirectory("${SIPROS_ROOT}/MSToolkit" "${CMAKE_BINARY_DIR}/vendor")
foreach(lib mstoolkit expat mzid mzparser zlib)
  set_target_properties(${lib} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endforeach()
set(ORIGINAL_SOURCES
  original/src/CometSearchMod.cpp original/src/MVH.cpp
  original/src/SiprosReader.cpp original/src/averagine.cpp
  original/src/isotopologue.cpp original/src/ms2scan.cpp
  original/src/peptide.cpp
  original/src/proNovoConfig.cpp original/src/proteindatabase.cpp
  original/src/ptm.cpp original/src/tokenvector.cpp)
add_library(mvh_original STATIC ${ORIGINAL_SOURCES}
  src/mvh_scan_vector.cpp src/spectrum_input.cpp src/database_search.cpp cuda/engine.cu cuda/score_impact.cu)
target_include_directories(mvh_original PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  "${CMAKE_CURRENT_SOURCE_DIR}/original/include")
target_link_libraries(mvh_original PUBLIC mstoolkit OpenMP::OpenMP_CXX CUDA::cudart)
if(MSVC)
  target_compile_options(mvh_original PRIVATE /fp:fast /openmp:experimental /openmp:llvm /arch:AVX2)
else()
  # Same floating-point/architecture options as upstream BUILD_CONDA=ON.
  target_compile_options(mvh_original PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffast-math;-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
endif()
if(WIN32)
  target_compile_definitions(mvh_original PUBLIC _NO_THERMORAW _NOSQLITE)
else()
  target_compile_definitions(mvh_original PUBLIC GCC _FILE_OFFSET_BITS=64 _NOSQLITE)
endif()
add_executable(sipros_mvh_cuda app/main.cpp app/runner.cpp)
target_include_directories(sipros_mvh_cuda PRIVATE "${SIPROS_ROOT}/shared")
target_compile_definitions(sipros_mvh_cuda PRIVATE SIPROS_PROJECT_ROOT="${SIPROS_ROOT}")
target_link_libraries(sipros_mvh_cuda PRIVATE mvh_original)
set_target_properties(sipros_mvh_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME mvh_original_source_identity
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_original.py")
add_test(NAME mvh_cli_and_sample
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_run.py"
          --binary $<TARGET_FILE:sipros_mvh_cuda> --root "${SIPROS_ROOT}")

add_test(NAME cuda_conversion_scope
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_conversion.py")

add_executable(mvh_cuda_contract tests/cuda_contract.cpp)
target_link_libraries(mvh_cuda_contract PRIVATE mvh_original)
set_target_properties(mvh_cuda_contract PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME cuda_semantic_contract
  COMMAND mvh_cuda_contract "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/search.cfg")

add_executable(score_impact_contract tests/score_impact_contract.cpp)
target_link_libraries(score_impact_contract PRIVATE mvh_original)
add_test(NAME score_impact_contract COMMAND score_impact_contract)

option(MVH_CUDA_ENABLE_RT "Enable experimental GPU-resident RT matching" OFF)
if(MVH_CUDA_ENABLE_RT)
  add_subdirectory("${SIPROS_ROOT}/MVH_RT/optix_example" "${CMAKE_CURRENT_BINARY_DIR}/rt_support")
  set(GPU_RT_DIR "${SIPROS_ROOT}/MVH_RT/gpu_bridge")
  set(GPU_RT_PTX "${CMAKE_CURRENT_BINARY_DIR}/gpu_rt.ptx")
  add_custom_command(OUTPUT "${GPU_RT_PTX}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --std=c++17 --gpu-architecture=compute_75
      --generate-line-info --fmad=false "-I${OPTIX_INCLUDE_DIR}"
      "-I${CMAKE_CURRENT_SOURCE_DIR}/cuda" "-I${GPU_RT_DIR}"
      "${GPU_RT_DIR}/device.cu" -o "${GPU_RT_PTX}"
    DEPENDS "${GPU_RT_DIR}/device.cu" "${GPU_RT_DIR}/bridge.h"
      cuda/types.cuh cuda/scoring.cuh VERBATIM)
  add_custom_target(mvh_gpu_rt_ptx DEPENDS "${GPU_RT_PTX}")
  set(GPU_RT_CUSTOM_PTX "${CMAKE_CURRENT_BINARY_DIR}/gpu_rt_custom.ptx")
  add_custom_command(OUTPUT "${GPU_RT_CUSTOM_PTX}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --std=c++17 --gpu-architecture=compute_75
      --generate-line-info --fmad=false "-I${OPTIX_INCLUDE_DIR}"
      "-I${CMAKE_CURRENT_SOURCE_DIR}/cuda" "-I${GPU_RT_DIR}"
      "${GPU_RT_DIR}/custom_device.cu" -o "${GPU_RT_CUSTOM_PTX}"
    DEPENDS "${GPU_RT_DIR}/custom_device.cu" "${GPU_RT_DIR}/bridge.h"
      cuda/types.cuh cuda/scoring.cuh VERBATIM)
  add_custom_target(mvh_gpu_rt_custom_ptx DEPENDS "${GPU_RT_CUSTOM_PTX}")
  add_library(mvh_gpu_rt STATIC "${GPU_RT_DIR}/bridge.cpp"
    "${GPU_RT_DIR}/sphere_backend.cpp" "${GPU_RT_DIR}/triangle_backend.cpp"
    "${GPU_RT_DIR}/scene_resources.cpp" "${GPU_RT_DIR}/geometry.cu"
    "${GPU_RT_DIR}/custom_geometry.cu")
  target_include_directories(mvh_gpu_rt PUBLIC "${GPU_RT_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
  target_link_libraries(mvh_gpu_rt PUBLIC mvh_rt_support)
  target_compile_definitions(mvh_gpu_rt PRIVATE MVH_GPU_RT_PTX_PATH="${GPU_RT_PTX}"
    MVH_GPU_RT_CUSTOM_PTX_PATH="${GPU_RT_CUSTOM_PTX}")
  add_dependencies(mvh_gpu_rt mvh_gpu_rt_ptx mvh_gpu_rt_custom_ptx)
  target_link_libraries(mvh_original PUBLIC mvh_gpu_rt)
  target_compile_definitions(mvh_original PRIVATE MVH_CUDA_ENABLE_RT=1)
  add_executable(mvh_gpu_rt_contract "${GPU_RT_DIR}/contract.cpp")
  target_link_libraries(mvh_gpu_rt_contract PRIVATE mvh_gpu_rt)
  add_test(NAME gpu_rt_contract COMMAND mvh_gpu_rt_contract)
  add_test(NAME gpu_rt_instanced_contract COMMAND mvh_gpu_rt_contract instanced)
  add_test(NAME gpu_rt_custom_contract COMMAND mvh_gpu_rt_contract custom)
  add_executable(mvh_gpu_rt_sphere_contract "${GPU_RT_DIR}/custom_contract.cpp")
  target_link_libraries(mvh_gpu_rt_sphere_contract PRIVATE mvh_gpu_rt)
  add_test(NAME gpu_rt_sphere_layout COMMAND mvh_gpu_rt_sphere_contract)
  add_test(NAME score_impact_end_to_end COMMAND ${Python3_EXECUTABLE} -B
    "${GPU_RT_DIR}/check_score_impact.py" --binary $<TARGET_FILE:sipros_mvh_cuda>)
  add_test(NAME score_impact_report_semantics COMMAND ${Python3_EXECUTABLE} -B
    "${GPU_RT_DIR}/test_score_impact_report.py")
  add_test(NAME gpu_rt_integration COMMAND ${Python3_EXECUTABLE}
    "${GPU_RT_DIR}/check_integration.py" --binary $<TARGET_FILE:sipros_mvh_cuda>
    --root "${SIPROS_ROOT}")

endif()

# Check default paths and explicit overrides against a real sample search.
add_test(NAME output_directory_contract
  COMMAND ${Python3_EXECUTABLE} -B "${SIPROS_ROOT}/shared/tests/check_output_paths.py"
          --binary $<TARGET_FILE:sipros_mvh_cuda> --component mvh_cuda)
