# benchmarks/CMakeLists.txt

# Prefer a system-installed Google Benchmark package and fall back to FetchContent.
find_package(benchmark CONFIG QUIET)

if(NOT benchmark_FOUND)
    include(FetchContent)

    # Optionally, include Google Test as a dependency for Google Benchmark's testing
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googlebenchmark
        URL      https://github.com/google/benchmark/archive/refs/tags/v1.9.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # Fetch and make available Google Benchmark
    FetchContent_MakeAvailable(googlebenchmark)
endif()

# Collect all benchmark source files
file(GLOB BENCHMARK_SOURCES "*.cpp")

# Add the benchmark executable
add_executable("${PROJECT_NAME}_benchmarks" ${BENCHMARK_SOURCES})

# Link the benchmark executable against bdf17_core and Google Benchmark
target_link_libraries("${PROJECT_NAME}_benchmarks"
    PRIVATE
    bdf17_core
    benchmark::benchmark_main
)

# Optionally, set compile options for benchmarks
target_compile_options("${PROJECT_NAME}_benchmarks" PRIVATE -Ofast -march=native -mavx2 -funroll-loops -ftree-vectorize -Wno-unused-parameter)

# Installation rules for benchmarks (optional)
# install(TARGETS gpv23_benchmarks DESTINATION bin)
