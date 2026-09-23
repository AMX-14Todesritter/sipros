cmake_minimum_required(VERSION 3.18)
project(SiprosMvhOriginal LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenMP REQUIRED)
get_filename_component(SIPROS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
# Shared vendor dependency; no duplicated MSToolkit checkout.
include_directories("${SIPROS_ROOT}/MSToolkit/include")
add_subdirectory("${SIPROS_ROOT}/MSToolkit" "${CMAKE_BINARY_DIR}/vendor")
foreach(lib mstoolkit expat mzid mzparser zlib)
  set_target_properties(${lib} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endforeach()
set(ORIGINAL_SOURCES
  original/src/CometSearchMod.cpp src/mvh_scoring.cpp
  original/src/SiprosReader.cpp original/src/averagine.cpp
  original/src/isotopologue.cpp original/src/ms2scan.cpp
  original/src/peptide.cpp
  original/src/proNovoConfig.cpp original/src/proteindatabase.cpp
  original/src/ptm.cpp original/src/tokenvector.cpp)
add_library(mvh_original STATIC ${ORIGINAL_SOURCES}
  src/mvh_scan_vector.cpp src/spectrum_input.cpp src/database_search.cpp)
target_include_directories(mvh_original PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/original/include")
target_link_libraries(mvh_original PUBLIC mstoolkit OpenMP::OpenMP_CXX)
if(MSVC)
  target_compile_options(mvh_original PRIVATE /fp:fast /openmp:experimental /openmp:llvm /arch:AVX2)
else()
  # Same floating-point/architecture options as upstream BUILD_CONDA=ON.
  target_compile_options(mvh_original PRIVATE -ffast-math -march=native)
endif()
if(WIN32)
  target_compile_definitions(mvh_original PUBLIC _NO_THERMORAW _NOSQLITE)
else()
  target_compile_definitions(mvh_original PUBLIC GCC _FILE_OFFSET_BITS=64 _NOSQLITE)
endif()
add_executable(sipros_mvh app/main.cpp app/runner.cpp)
target_include_directories(sipros_mvh PRIVATE "${SIPROS_ROOT}/shared")
target_compile_definitions(sipros_mvh PRIVATE SIPROS_PROJECT_ROOT="${SIPROS_ROOT}")
target_link_libraries(sipros_mvh PRIVATE mvh_original)
option(MVH_ENABLE_RT "Build RT resources after MVH preprocessing" OFF)

if(MVH_ENABLE_RT)
  set(MVH_RT_TEST_ARGS --expect-rt)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../optix_example"
    "${CMAKE_CURRENT_BINARY_DIR}/rt"
  )

  target_link_libraries(sipros_mvh PRIVATE mvh_rt_support)
  add_dependencies(sipros_mvh optix_device_programs)

  target_compile_definitions(sipros_mvh PRIVATE
    MVH_ENABLE_RT=1
    MVH_RT_PTX_PATH="${CMAKE_CURRENT_BINARY_DIR}/rt/device_programs.ptx"
  )
endif()
set_target_properties(sipros_mvh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME mvh_original_source_identity
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_original.py")
add_test(NAME mvh_cli_and_sample
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_run.py"
          --binary $<TARGET_FILE:sipros_mvh> --root "${SIPROS_ROOT}" ${MVH_RT_TEST_ARGS})

add_test(NAME mvh_scoring_observer_scope
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_scoring_observer.py")

add_test(NAME mvh_extracted_methods_identity
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_extraction.py")

# Check default paths and explicit overrides against a real sample search.
add_test(NAME output_directory_contract
  COMMAND ${Python3_EXECUTABLE} -B "${SIPROS_ROOT}/shared/tests/check_output_paths.py"
          --binary $<TARGET_FILE:sipros_mvh> --component mvh_rt)
