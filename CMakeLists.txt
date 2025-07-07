cmake_minimum_required(VERSION 3.10)
project(SQPnP_Clean VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Print project info
message(STATUS "=== SQPnP Clean Build ===")
message(STATUS "Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen3 Found: ${EIGEN3_FOUND}")
message(STATUS "Eigen3 Include: ${EIGEN3_INCLUDE_DIRS}")

# Add subdirectories
add_subdirectory(sqpnp)

# Create the original clean example executable
add_executable(clean_example clean_example.cpp)

# Link against SQPnP library and Eigen3
target_link_libraries(clean_example SQPnP Eigen3::Eigen)

# Set include directories
target_include_directories(clean_example PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

# Add compiler flags
target_compile_options(clean_example PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /O2>
)

# Create the command-line interface executable
add_executable(sqpnp_cli clean_example_cli.cpp)

# Link against SQPnP library and Eigen3
target_link_libraries(sqpnp_cli SQPnP Eigen3::Eigen)

# Set include directories
target_include_directories(sqpnp_cli PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

# Add compiler flags
target_compile_options(sqpnp_cli PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /O2>
)

# Print build info
message(STATUS "Targets:")
message(STATUS "  - SQPnP (library)")
message(STATUS "  - clean_example (original with hardcoded data)")
message(STATUS "  - sqpnp_cli (command-line interface)")
message(STATUS "")
message(STATUS "Build with: cmake .. && make")
message(STATUS "Run examples:")
message(STATUS "  ./clean_example                    # Original with hardcoded data")
message(STATUS "  ./sqpnp_cli --demo                # CLI with demo data")
message(STATUS "  ./sqpnp_cli -d data.txt -c K.txt  # CLI with real data")
message(STATUS "  ./sqpnp_cli --help                # Show CLI help") 