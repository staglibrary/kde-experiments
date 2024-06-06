cmake_minimum_required(VERSION 3.14)
project(kde_experiments)

# Require at least C++20
set(CMAKE_CXX_STANDARD 20)

# Add the cmake folder
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Default to release build if CMAKE_BUILD_TYPE is not configured
if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "[kde] Defaulting to release build")
    set(CMAKE_BUILD_TYPE "Release")
else()
    message(STATUS "[kde] Build type: ${CMAKE_BUILD_TYPE}")
endif()

# Find and include the STAG library
find_package(stag REQUIRED)
message(STATUS "[kde] Found STAG!")
include_directories(${STAG_INCLUDE_DIRS})

# Find and include the HDF5 library
list(APPEND CMAKE_PREFIX_PATH "/usr/local/hdf5/")
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/share/HighFive/")
find_package(HighFive REQUIRED)

add_executable(
        kde_experiments EXCLUDE_FROM_ALL
        src/main.cpp
        src/data.h
        src/data.cpp
        src/logging.h
        src/experiments.cpp
        src/experiments.h)

target_link_libraries(kde_experiments HighFive)
target_link_libraries(kde_experiments stag)

