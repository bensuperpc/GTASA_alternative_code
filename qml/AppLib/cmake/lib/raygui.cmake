cmake_minimum_required(VERSION 3.14.0)

include(FetchContent)

set(BUILD_RAYLIB_CPP_EXAMPLES OFF CACHE BOOL "" FORCE)

find_package(raygui QUIET)

if (NOT raygui_FOUND)
    FetchContent_Declare(raygui
        GIT_REPOSITORY https://github.com/raysan5/raygui.git
        GIT_TAG 4.0
    )
    FetchContent_MakeAvailable(raygui)
    include_directories(SYSTEM ${raygui_SOURCE_DIR})
    include_directories(SYSTEM ${raygui_SOURCE_DIR}/src)
endif()