cmake_minimum_required(VERSION 3.16)
project(sqpnp_python_bindings)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Build SQPnP library inline
message(STATUS "Building SQPnP library inline...")

# Create SQPnP library
add_library(SQPnP STATIC ${CMAKE_SOURCE_DIR}/../sqpnp/sqpnp.cpp)

# Set properties for SQPnP library (keep C++14 for compatibility)
set_target_properties(SQPnP PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

# Include directories for SQPnP
target_include_directories(SQPnP PUBLIC
    ${CMAKE_SOURCE_DIR}/../sqpnp
    ${EIGEN3_INCLUDE_DIR}
)

# Compiler flags for SQPnP
target_compile_options(SQPnP PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /O2>
)

# Create the Python module using pybind11
pybind11_add_module(sqpnp_python sqpnp_bindings.cpp)

# Set the output name to match Python's expected format
set_target_properties(sqpnp_python PROPERTIES
    OUTPUT_NAME "sqpnp_python"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Include directories
target_include_directories(sqpnp_python PRIVATE 
    ${CMAKE_SOURCE_DIR}/../sqpnp
    ${EIGEN3_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(sqpnp_python PRIVATE 
    SQPnP 
    Eigen3::Eigen
)

# Print status
message(STATUS "Building SQPnP Python bindings with pybind11")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "SQPnP source: ${CMAKE_SOURCE_DIR}/../sqpnp/sqpnp.cpp")