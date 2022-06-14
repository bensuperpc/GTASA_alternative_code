cmake_minimum_required(VERSION 3.14.0)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 191ca1f3a9262b90a586ae2c2e8c742c3d867801) # 14-06-2022

# Disable tests on gtest
set(gtest_build_tests
    OFF
    CACHE BOOL "" FORCE)
set(gtest_build_samples
    OFF
    CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
# Lib: gtest gtest_main
