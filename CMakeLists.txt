cmake_minimum_required(VERSION 3.24)
project(bdf17 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(CTest)

option(ENABLE_BENCHMARKS "Enable benchmarking with Google Benchmark" ON)
option(ENABLE_AVX2 "Enable AVX2 NTT kernels (requires AVX2-capable toolchain/CPU)" OFF)
option(ENABLE_AVX512 "Enable AVX-512 NTT kernels (requires AVX-512 capable toolchain/CPU)" OFF)
option(BDF17_PORTABLE "Disable host-specific -march tuning for portable builds" ON)
option(BDF17_NATIVE_ARCH "Enable -march=native for local performance builds" OFF)
option(BDF17_ENABLE_OPENMP "Enable OpenMP support when available" ON)
option(BDF17_FETCH_DEPS "Allow FetchContent to download optional dependencies" ON)
option(BDF17_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(BDF17_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(BDF17_RUNTIME_BACKEND_DISPATCH "Enable runtime backend dispatch for Backend::Auto" OFF)

set(BDF17_ARCH_FLAGS)
if(BDF17_PORTABLE)
    if(BDF17_NATIVE_ARCH)
        message(WARNING "BDF17_PORTABLE=ON ignores BDF17_NATIVE_ARCH=ON")
    endif()
elseif(BDF17_NATIVE_ARCH)
    list(APPEND BDF17_ARCH_FLAGS -march=native)
endif()

if(ENABLE_AVX2)
    list(APPEND BDF17_ARCH_FLAGS -mavx2)
endif()
if(ENABLE_AVX512)
    list(APPEND BDF17_ARCH_FLAGS
        -mavx512f
        -mavx512dq
    )
endif()

set(BDF17_COMPILE_DEFS)
if(ENABLE_AVX2)
    list(APPEND BDF17_COMPILE_DEFS BDF17_ENABLE_AVX2=1)
endif()
if(ENABLE_AVX512)
    list(APPEND BDF17_COMPILE_DEFS BDF17_ENABLE_AVX512=1)
endif()
if(BDF17_RUNTIME_BACKEND_DISPATCH)
    list(APPEND BDF17_COMPILE_DEFS BDF17_ENABLE_RUNTIME_BACKEND_DISPATCH=1)
endif()

set(BDF17_SANITIZER_FLAGS)
if(BDF17_ENABLE_ASAN)
    list(APPEND BDF17_SANITIZER_FLAGS -fsanitize=address)
endif()
if(BDF17_ENABLE_UBSAN)
    list(APPEND BDF17_SANITIZER_FLAGS -fsanitize=undefined)
endif()
if(BDF17_SANITIZER_FLAGS AND MSVC)
    message(FATAL_ERROR "Sanitizer options are currently only supported with GCC/Clang toolchains")
endif()

if(BDF17_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(NOT OpenMP_CXX_FOUND)
        message(STATUS "OpenMP requested but not found; building without OpenMP")
    endif()
endif()

set(BDF17_WARNING_FLAGS
    -Wall
    -Wextra
    -pedantic
    -Werror
    -Wno-unused-parameter
)

set(BDF17_PERF_FLAGS
    $<$<CONFIG:Release>:-O3>
    -funroll-loops
    -ftree-vectorize
)

function(bdf17_apply_build_settings target scope)
    target_include_directories(${target} ${scope} ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(${target} ${scope}
        ${BDF17_WARNING_FLAGS}
        ${BDF17_PERF_FLAGS}
        ${BDF17_ARCH_FLAGS}
    )

    if(BDF17_COMPILE_DEFS)
        target_compile_definitions(${target} ${scope} ${BDF17_COMPILE_DEFS})
    endif()

    if(BDF17_SANITIZER_FLAGS)
        target_compile_options(${target} ${scope}
            ${BDF17_SANITIZER_FLAGS}
            -fno-omit-frame-pointer
        )
        target_link_options(${target} ${scope} ${BDF17_SANITIZER_FLAGS})
    endif()

    if(BDF17_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(${target} ${scope} OpenMP::OpenMP_CXX)
    endif()
endfunction()

file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp")
set(ENTRYPOINT_SOURCE "${PROJECT_SOURCE_DIR}/src/main.cpp")
set(CORE_SOURCES ${PROJECT_SOURCES})
list(REMOVE_ITEM CORE_SOURCES "${ENTRYPOINT_SOURCE}")

if(CORE_SOURCES)
    add_library(bdf17_core)
    target_sources(bdf17_core PRIVATE ${CORE_SOURCES})
    bdf17_apply_build_settings(bdf17_core PUBLIC)
else()
    add_library(bdf17_core INTERFACE)
    bdf17_apply_build_settings(bdf17_core INTERFACE)
endif()

add_executable(${PROJECT_NAME} ${ENTRYPOINT_SOURCE})
set_target_properties(${PROJECT_NAME} PROPERTIES ENABLE_EXPORTS ON)
target_link_libraries(${PROJECT_NAME} PRIVATE bdf17_core)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation rules (optional)
# install(TARGETS ${PROJECT_NAME} DESTINATION bin)
