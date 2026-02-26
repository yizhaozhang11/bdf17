# tests/CMakeLists.txt

# Prefer a system-installed GoogleTest package and fall back to FetchContent.
find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL      https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # Keep GoogleTest's options from leaking into parent projects.
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)
endif()

if(TARGET GTest::gtest_main)
    set(BDF17_GTEST_MAIN_TARGET GTest::gtest_main)
else()
    set(BDF17_GTEST_MAIN_TARGET gtest_main)
endif()

# Collect all test source files
file(GLOB TEST_SOURCES "*.cpp")

# Add the test executable
add_executable("${PROJECT_NAME}_tests" ${TEST_SOURCES})

# Link the test executable against bdf17_core and gtest_main
target_link_libraries("${PROJECT_NAME}_tests"
    PRIVATE
    bdf17_core
    ${BDF17_GTEST_MAIN_TARGET}
)

# Add test cases to CTest
include(GoogleTest)
gtest_discover_tests("${PROJECT_NAME}_tests")
