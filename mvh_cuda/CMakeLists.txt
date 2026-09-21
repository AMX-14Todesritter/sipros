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
  src/mvh_scan_vector.cpp src/spectrum_input.cpp src/database_search.cpp cuda/engine.cu)
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
