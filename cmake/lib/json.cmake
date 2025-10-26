cmake_minimum_required(VERSION 3.14.0)

find_package(nlohmann_json QUIET)

if (NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found on system, downloading...")
    include(FetchContent)

    #set(CMAKE_MODULE_PATH
    #    ""
    #    CACHE STRING "" FORCE)

    #set(NLOHMANN_JSON_SYSTEM_INCLUDE
    #    ""
    #    CACHE STRING "" FORCE)

    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3 # 2023-11-28
    )
    FetchContent_MakeAvailable(nlohmann_json)
    # nlohmann_json::nlohmann_json
    set_target_properties(nlohmann_json
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    include_directories(${nlohmann_json_SOURCE_DIR}/include)
endif()
