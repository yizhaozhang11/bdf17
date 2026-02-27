cmake_minimum_required(VERSION 3.24)
project(bdf17 VERSION 1.0.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Optionally set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optionally enable benchmarks
option(ENABLE_BENCHMARKS "Enable benchmarking with Google Benchmark" ON)
option(ENABLE_AVX512 "Enable AVX-512 NTT kernels (requires AVX-512 capable toolchain/CPU)" OFF)

set(BDF17_ARCH_FLAGS
    -march=native
    -mavx2
)
if(ENABLE_AVX512)
    list(APPEND BDF17_ARCH_FLAGS
        -mavx512f
        -mavx512dq
    )
endif()

# Add source files
file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp")
set(ENTRYPOINT_SOURCE "${PROJECT_SOURCE_DIR}/src/main.cpp")
set(CORE_SOURCES ${PROJECT_SOURCES})
list(REMOVE_ITEM CORE_SOURCES "${ENTRYPOINT_SOURCE}")

# Core library target for non-entrypoint code and shared build settings
if(CORE_SOURCES)
    add_library(bdf17_core)
    target_sources(bdf17_core PRIVATE ${CORE_SOURCES})
    target_include_directories(bdf17_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(bdf17_core PUBLIC
        -Wall
        -Wextra
        -pedantic
        -Werror
        -Ofast
        ${BDF17_ARCH_FLAGS}
        -funroll-loops
        -ftree-vectorize
        -fopenmp
        -Wno-unused-parameter
    )
else()
    # Support single-file layouts where only the executable entrypoint exists.
    add_library(bdf17_core INTERFACE)
    target_include_directories(bdf17_core INTERFACE ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(bdf17_core INTERFACE
        -Wall
        -Wextra
        -pedantic
        -Werror
        -Ofast
        ${BDF17_ARCH_FLAGS}
        -funroll-loops
        -ftree-vectorize
        -fopenmp
        -Wno-unused-parameter
    )
endif()

# Add executable entrypoint
add_executable(${PROJECT_NAME} ${ENTRYPOINT_SOURCE})
set_target_properties(${PROJECT_NAME} PROPERTIES ENABLE_EXPORTS ON)
target_link_libraries(${PROJECT_NAME} PRIVATE bdf17_core)

# Enable testing
enable_testing()

# Add the tests subdirectory
add_subdirectory(tests)

if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation rules (optional)
# install(TARGETS ${PROJECT_NAME} DESTINATION bin)
