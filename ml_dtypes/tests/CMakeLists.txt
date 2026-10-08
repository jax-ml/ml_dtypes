cmake_minimum_required(VERSION 3.14)
project(my_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GOOGLETEST_DOWNLOAD_URL https://github.com/google/googletest/archive/refs/tags/v1.12.0.zip)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL ${GOOGLETEST_DOWNLOAD_URL}
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_GOOGLETEST_DOWNLOAD_URL ${GOOGLETEST_DOWNLOAD_URL})
add_subdirectory(abseil-cpp)

enable_testing()

add_executable(
  float8_test
  float8_test.cc
)
target_include_directories(float8_test PUBLIC
  ..
  ../..
  ../../third_party/eigen
)

target_link_libraries(
  float8_test
  GTest::gtest_main
  GTest::gmock_main
  absl::strings
)

include(GoogleTest)
gtest_discover_tests(float8_test)
