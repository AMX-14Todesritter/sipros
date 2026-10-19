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
add_library(mvh_matching STATIC matching/matching.cu)
target_include_directories(mvh_matching PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  "${CMAKE_CURRENT_SOURCE_DIR}/matching")
target_link_libraries(mvh_matching PUBLIC CUDA::cudart)
target_compile_options(mvh_matching PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
add_library(mvh_original STATIC ${ORIGINAL_SOURCES}
  src/mvh_scan_vector.cpp src/spectrum_input.cpp src/database_search.cpp cuda/engine.cu cuda/score_impact.cu cuda/peptide_generation.cu src/generation_mass.cpp)
target_include_directories(mvh_original PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  "${CMAKE_CURRENT_SOURCE_DIR}/original/include")
target_link_libraries(mvh_original PUBLIC mstoolkit OpenMP::OpenMP_CXX CUDA::cudart mvh_matching)
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
      cuda/types.cuh cuda/scoring.cuh cuda/ion_math.cuh matching/fusion.cuh matching/matching.cuh matching/diagnostics.cuh
      "${GPU_RT_DIR}/sphere_matching.cuh" VERBATIM)
  add_custom_target(mvh_gpu_rt_ptx DEPENDS "${GPU_RT_PTX}")
  set(GPU_RT_CUSTOM_PTX "${CMAKE_CURRENT_BINARY_DIR}/gpu_rt_custom.ptx")
  add_custom_command(OUTPUT "${GPU_RT_CUSTOM_PTX}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --std=c++17 --gpu-architecture=compute_75
      --generate-line-info --fmad=false "-I${OPTIX_INCLUDE_DIR}"
      "-I${CMAKE_CURRENT_SOURCE_DIR}/cuda" "-I${GPU_RT_DIR}"
      "${GPU_RT_DIR}/custom_device.cu" -o "${GPU_RT_CUSTOM_PTX}"
    DEPENDS "${GPU_RT_DIR}/custom_device.cu" "${GPU_RT_DIR}/bridge.h"
      cuda/types.cuh cuda/scoring.cuh cuda/ion_math.cuh matching/fusion.cuh matching/matching.cuh matching/diagnostics.cuh
      "${GPU_RT_DIR}/sphere_matching.cuh" VERBATIM)
  add_custom_target(mvh_gpu_rt_custom_ptx DEPENDS "${GPU_RT_CUSTOM_PTX}")
  foreach(variant IN ITEMS aggregate aggregate_diagnostics)
    set(AGGREGATE_PTX_${variant} "${CMAKE_CURRENT_BINARY_DIR}/gpu_rt_${variant}.ptx")
    set(AGGREGATE_DEFINES -DMVH_MATCHING_AGGREGATE=1)
    if(variant STREQUAL "aggregate_diagnostics")
      list(APPEND AGGREGATE_DEFINES -DMVH_MATCHING_DIAGNOSTICS=1)
    endif()
    add_custom_command(OUTPUT "${AGGREGATE_PTX_${variant}}"
      COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --std=c++17 --gpu-architecture=compute_75
        --generate-line-info --fmad=false ${AGGREGATE_DEFINES}
        "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/cuda" "-I${GPU_RT_DIR}"
        "${GPU_RT_DIR}/custom_device.cu" -o "${AGGREGATE_PTX_${variant}}"
      DEPENDS "${GPU_RT_DIR}/custom_device.cu" "${GPU_RT_DIR}/bridge.h"
        "${GPU_RT_DIR}/sphere_matching.cuh" cuda/types.cuh cuda/scoring.cuh
        cuda/ion_math.cuh matching/fusion.cuh matching/matching.cuh matching/diagnostics.cuh VERBATIM)
    add_custom_target(mvh_gpu_rt_${variant}_ptx DEPENDS "${AGGREGATE_PTX_${variant}}")
  endforeach()

  foreach(mode RANGE 0 5)
    math(EXPR FUSION_ALGORITHM_MODE "${mode} % 3")
    set(FUSION_POSTCHECK_DEFINES)
    if(mode GREATER_EQUAL 3)
      list(APPEND FUSION_POSTCHECK_DEFINES -DMVH_FUSION_RT_NO_POSTCHECK=1)
    endif()
    set(FUSION_PTX_${mode} "${CMAKE_CURRENT_BINARY_DIR}/gpu_rt_fusion_${mode}.ptx")
    add_custom_command(OUTPUT "${FUSION_PTX_${mode}}"
      COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --std=c++17 --gpu-architecture=compute_75
        --generate-line-info --fmad=false -DMVH_FUSION_MODE=${FUSION_ALGORITHM_MODE} ${FUSION_POSTCHECK_DEFINES}
        "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/cuda" "-I${GPU_RT_DIR}"
        "${GPU_RT_DIR}/custom_device.cu" -o "${FUSION_PTX_${mode}}"
      DEPENDS "${GPU_RT_DIR}/custom_device.cu" "${GPU_RT_DIR}/bridge.h"
        "${GPU_RT_DIR}/sphere_matching.cuh" cuda/types.cuh cuda/scoring.cuh
        cuda/ion_math.cuh matching/fusion.cuh matching/matching.cuh matching/diagnostics.cuh VERBATIM)
    add_custom_target(mvh_gpu_rt_fusion_${mode}_ptx DEPENDS "${FUSION_PTX_${mode}}")
  endforeach()

  add_library(mvh_gpu_rt STATIC "${GPU_RT_DIR}/bridge.cpp"
    "${GPU_RT_DIR}/sphere_backend.cpp" "${GPU_RT_DIR}/triangle_backend.cpp"
    "${GPU_RT_DIR}/scene_resources.cpp" "${GPU_RT_DIR}/geometry.cu"
    "${GPU_RT_DIR}/custom_geometry.cu")
  target_include_directories(mvh_gpu_rt PUBLIC "${GPU_RT_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
  target_link_libraries(mvh_gpu_rt PUBLIC mvh_rt_support)
  target_compile_definitions(mvh_gpu_rt PRIVATE MVH_GPU_RT_PTX_PATH="${GPU_RT_PTX}"
    MVH_GPU_RT_CUSTOM_PTX_PATH="${GPU_RT_CUSTOM_PTX}"
    MVH_GPU_RT_AGGREGATE_PTX_PATH="${AGGREGATE_PTX_aggregate}")
  add_dependencies(mvh_gpu_rt mvh_gpu_rt_ptx mvh_gpu_rt_custom_ptx mvh_gpu_rt_aggregate_ptx)
  target_compile_definitions(mvh_gpu_rt PRIVATE
    MVH_GPU_RT_FUSION_0_PTX_PATH="${FUSION_PTX_0}"
    MVH_GPU_RT_FUSION_1_PTX_PATH="${FUSION_PTX_1}"
    MVH_GPU_RT_FUSION_2_PTX_PATH="${FUSION_PTX_2}"
    MVH_GPU_RT_FUSION_3_PTX_PATH="${FUSION_PTX_3}"
    MVH_GPU_RT_FUSION_4_PTX_PATH="${FUSION_PTX_4}"
    MVH_GPU_RT_FUSION_5_PTX_PATH="${FUSION_PTX_5}")
  add_dependencies(mvh_gpu_rt mvh_gpu_rt_fusion_0_ptx mvh_gpu_rt_fusion_1_ptx mvh_gpu_rt_fusion_2_ptx mvh_gpu_rt_fusion_3_ptx mvh_gpu_rt_fusion_4_ptx mvh_gpu_rt_fusion_5_ptx)
  add_executable(matching_fusion_replay matching/fusion_replay.cu)
  target_link_libraries(matching_fusion_replay PRIVATE mvh_matching mvh_gpu_rt OpenMP::OpenMP_CXX)
  target_compile_options(matching_fusion_replay PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo;-Xcompiler=-fopenmp;-Xcompiler=-ffp-contract=off>)
  set_target_properties(matching_fusion_replay PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  add_executable(matching_fusion_contract tests/fusion_contract.cu)
  target_link_libraries(matching_fusion_contract PRIVATE mvh_matching)
  target_compile_options(matching_fusion_contract PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-Xcompiler=-ffp-contract=off>)
  add_test(NAME matching_fusion_original_scoring_contract COMMAND matching_fusion_contract)
  add_test(NAME matching_fusion_replay_contract COMMAND ${Python3_EXECUTABLE} -B
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_fusion_replay.py"
    --search $<TARGET_FILE:sipros_mvh_cuda> --replay $<TARGET_FILE:matching_fusion_replay>
    --root "${SIPROS_ROOT}")
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

add_executable(sequence_ids_contract tests/sequence_ids_contract.cpp)
target_include_directories(sequence_ids_contract PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_test(NAME sequence_ids_contract COMMAND sequence_ids_contract)

add_executable(sequence_ids_reuse_contract tests/sequence_ids_reuse_contract.cpp)
target_include_directories(sequence_ids_reuse_contract PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_test(NAME sequence_ids_reuse_contract COMMAND sequence_ids_reuse_contract)

add_executable(peptide_batch_contract tests/peptide_batch_contract.cpp)
target_include_directories(peptide_batch_contract PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_test(NAME peptide_batch_contract COMMAND peptide_batch_contract)

add_executable(generation_contract tests/generation_contract.cpp)
target_link_libraries(generation_contract PRIVATE mvh_original)
add_test(NAME cuda_generation_contract COMMAND ${Python3_EXECUTABLE} -B
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_generation.py" --binary $<TARGET_FILE:generation_contract>)

add_executable(instrument_precursor_contract tests/instrument_precursor_contract.cpp)
target_link_libraries(instrument_precursor_contract PRIVATE mvh_original)
add_test(NAME instrument_precursor_contract COMMAND instrument_precursor_contract
  "${SIPROS_ROOT}/experiments/Regular.cfg" "${CMAKE_BINARY_DIR}")

add_executable(matching_benchmark matching/benchmark.cpp)
target_link_libraries(matching_benchmark PRIVATE mvh_matching OpenMP::OpenMP_CXX)
if(MVH_CUDA_ENABLE_RT)
  target_compile_definitions(matching_benchmark PRIVATE MATCHING_ENABLE_RT=1)
  target_link_libraries(matching_benchmark PRIVATE mvh_gpu_rt)
  add_executable(matching_replay matching/replay.cpp)
  target_link_libraries(matching_replay PRIVATE mvh_matching mvh_gpu_rt)
  set_target_properties(matching_replay PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

  # Exhaustive CUDA control, linked independently to retain the bin baseline.
  add_library(mvh_matching_linear STATIC matching/matching.cu)
  target_include_directories(mvh_matching_linear PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda" "${CMAKE_CURRENT_SOURCE_DIR}/matching")
  target_link_libraries(mvh_matching_linear PUBLIC CUDA::cudart)
  target_compile_definitions(mvh_matching_linear PRIVATE MVH_MATCHING_LINEAR=1)
  target_compile_options(mvh_matching_linear PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
  add_executable(matching_replay_no_bin matching/replay.cpp)
  target_compile_definitions(matching_replay_no_bin PRIVATE MVH_MATCHING_LINEAR=1)
  target_link_libraries(matching_replay_no_bin PRIVATE mvh_matching_linear mvh_gpu_rt)
  set_target_properties(matching_replay_no_bin PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

  # Same algorithm sources, separate instrumented objects/PTX; never link both
  # normal and instrumented libraries into one executable.
  set(GPU_RT_DIAGNOSTIC_PTX "${CMAKE_CURRENT_BINARY_DIR}/gpu_rt_diagnostics.ptx")
  add_custom_command(OUTPUT "${GPU_RT_DIAGNOSTIC_PTX}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --std=c++17 --gpu-architecture=compute_75
      --generate-line-info --fmad=false -DMVH_MATCHING_DIAGNOSTICS=1
      "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/cuda" "-I${GPU_RT_DIR}"
      "${GPU_RT_DIR}/custom_device.cu" -o "${GPU_RT_DIAGNOSTIC_PTX}"
    DEPENDS "${GPU_RT_DIR}/custom_device.cu" "${GPU_RT_DIR}/bridge.h"
      "${GPU_RT_DIR}/sphere_matching.cuh" cuda/scoring.cuh cuda/types.cuh
      cuda/ion_math.cuh matching/fusion.cuh matching/matching.cuh matching/diagnostics.cuh VERBATIM)
  add_custom_target(mvh_gpu_rt_diagnostic_ptx DEPENDS "${GPU_RT_DIAGNOSTIC_PTX}")
  get_target_property(DIAGNOSTIC_RT_SOURCES mvh_gpu_rt SOURCES)
  add_library(mvh_gpu_rt_diagnostics STATIC ${DIAGNOSTIC_RT_SOURCES})
  target_include_directories(mvh_gpu_rt_diagnostics PUBLIC "${GPU_RT_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
  target_link_libraries(mvh_gpu_rt_diagnostics PUBLIC mvh_rt_support)
  target_compile_definitions(mvh_gpu_rt_diagnostics PRIVATE MVH_GPU_RT_PTX_PATH="${GPU_RT_PTX}"
    MVH_GPU_RT_CUSTOM_PTX_PATH="${GPU_RT_DIAGNOSTIC_PTX}"
    MVH_GPU_RT_AGGREGATE_PTX_PATH="${AGGREGATE_PTX_aggregate_diagnostics}")
  add_dependencies(mvh_gpu_rt_diagnostics mvh_gpu_rt_ptx mvh_gpu_rt_diagnostic_ptx mvh_gpu_rt_aggregate_diagnostics_ptx)
  add_library(mvh_matching_diagnostics STATIC matching/matching.cu)
  target_include_directories(mvh_matching_diagnostics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cuda" "${CMAKE_CURRENT_SOURCE_DIR}/matching")
  target_link_libraries(mvh_matching_diagnostics PUBLIC CUDA::cudart)
  target_compile_definitions(mvh_matching_diagnostics PRIVATE MVH_MATCHING_DIAGNOSTICS=1)
  target_compile_options(mvh_matching_diagnostics PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
  add_executable(matching_replay_diagnostics matching/replay.cpp)
  target_compile_definitions(matching_replay_diagnostics PRIVATE MVH_MATCHING_DIAGNOSTICS=1)
  target_link_libraries(matching_replay_diagnostics PRIVATE mvh_matching_diagnostics mvh_gpu_rt_diagnostics ${CMAKE_DL_LIBS})
  find_path(MATCHING_NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
  if(MATCHING_NVTX_INCLUDE_DIR)
    target_include_directories(matching_replay_diagnostics PRIVATE "${MATCHING_NVTX_INCLUDE_DIR}")
    target_compile_definitions(matching_replay_diagnostics PRIVATE MVH_MATCHING_NVTX=1)
    add_executable(matching_replay_nvtx matching/replay.cpp)
    target_include_directories(matching_replay_nvtx PRIVATE "${MATCHING_NVTX_INCLUDE_DIR}")
    target_compile_definitions(matching_replay_nvtx PRIVATE MVH_MATCHING_NVTX=1)
    target_link_libraries(matching_replay_nvtx PRIVATE mvh_matching mvh_gpu_rt ${CMAKE_DL_LIBS})
    set_target_properties(matching_replay_nvtx PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  else()
    message(WARNING "NVTX3 headers unavailable: matching counters enabled without NVTX ranges")
  endif()
  set_target_properties(matching_replay_diagnostics PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  if(TARGET matching_replay_nvtx)
    set(MATCHING_NVTX_TEST_ARGS --nvtx $<TARGET_FILE:matching_replay_nvtx>)
  endif()
  add_test(NAME matching_diagnostics_contract COMMAND ${Python3_EXECUTABLE} -B
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_matching_diagnostics.py"
    --normal $<TARGET_FILE:matching_replay>
    --diagnostics $<TARGET_FILE:matching_replay_diagnostics> --root "${SIPROS_ROOT}" ${MATCHING_NVTX_TEST_ARGS})

  add_test(NAME matching_no_bin_replay_contract COMMAND ${Python3_EXECUTABLE} -B
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_matching_capture.py"
    --binary $<TARGET_FILE:sipros_mvh_cuda> --root "${SIPROS_ROOT}"
    --replay $<TARGET_FILE:matching_replay_no_bin>)

  add_test(NAME matching_grouped_replay_contract COMMAND ${Python3_EXECUTABLE} -B
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_matching_capture.py"
    --binary $<TARGET_FILE:sipros_mvh_cuda> --root "${SIPROS_ROOT}"
    --replay $<TARGET_FILE:matching_replay>)
endif()
set_target_properties(matching_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME matching_module_contract COMMAND matching_benchmark --contract)
add_test(NAME matching_capture_contract COMMAND ${Python3_EXECUTABLE} -B
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_matching_capture.py"
  --binary $<TARGET_FILE:sipros_mvh_cuda> --root "${SIPROS_ROOT}")
