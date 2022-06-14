cmake_minimum_required(VERSION 3.14.0)

include(FetchContent)

set(BUILD_RAYLIB_CPP_EXAMPLES OFF CACHE BOOL "" FORCE)

find_package(json QUIET)

if (NOT json_FOUND)
    FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG 6b97599a274b9b72caffa1332d5384c9aac27590 # 2022-05-31
    )
    FetchContent_MakeAvailable(json)
    # nlohmann_json::nlohmann_json
endif()