cmake_minimum_required(VERSION 3.12)
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
  original/src/CometSearchMod.cpp original/src/MVH.cpp
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
target_link_libraries(sipros_mvh PRIVATE mvh_original)
set_target_properties(sipros_mvh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME mvh_original_source_identity
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_original.py")
add_test(NAME mvh_cli_and_sample
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_run.py"
          --binary $<TARGET_FILE:sipros_mvh> --root "${SIPROS_ROOT}")

add_test(NAME mvh_extracted_methods_identity
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_extraction.py")
