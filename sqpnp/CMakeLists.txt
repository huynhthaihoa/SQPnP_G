# SQPnP Library with integrated RANSAC
add_library(SQPnP 
    sqpnp.cpp
)

# Set C++14 standard
set_target_properties(SQPnP PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Include directories
target_include_directories(SQPnP PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# Compiler flags
target_compile_options(SQPnP PRIVATE -Wall -O3)

# Print information
message(STATUS "Building SQPnP with integrated RANSAC")
message(STATUS "Eigen3 found at: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Features: Simple PnP + Robust PnP with RANSAC")
message(STATUS "Dependencies: Eigen3 only") 