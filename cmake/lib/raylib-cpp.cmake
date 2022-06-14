cmake_minimum_required(VERSION 3.14.0)

include(FetchContent)

find_package(raylib_cpp QUIET)

if (NOT raylib_cpp_FOUND)
    FetchContent_Declare(raylib_cpp
        GIT_REPOSITORY https://github.com/RobLoach/raylib-cpp.git
        GIT_TAG e7731d306ead3b5b9dd1703d5ee8f4341a7a695c # 2022-05-31
    )
    FetchContent_MakeAvailable(raylib_cpp)
endif()