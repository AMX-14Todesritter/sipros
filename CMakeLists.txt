cmake_minimum_required(VERSION 3.12)
project(Sipros)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message("${CMAKE_BUILD_TYPE}")

message("${CMAKE_CXX_COMPILER}")
set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/MSToolkit/include)

SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

option(SIPROS_MVH_ONLY_BUILD "Configure search targets without MPI and workflow tools" OFF)
add_subdirectory(MSToolkit)
if(SIPROS_MVH_ONLY_BUILD)
  foreach(MVH_LIBRARY mstoolkit expat mzid mzparser zlib)
    set_target_properties(${MVH_LIBRARY} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  endforeach()
endif()
add_subdirectory(openmp)
add_subdirectory(mvh)
if(NOT SIPROS_MVH_ONLY_BUILD)
add_subdirectory(mpi)
add_subdirectory(AerithFeatureExtractor)
add_subdirectory(configGenerator)
endif()
