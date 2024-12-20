cmake_minimum_required(VERSION 3.14.0)

find_package(SortFilterProxyModel QUIET)

if (NOT SortFilterProxyModel_FOUND)
    message(STATUS "SortFilterProxyModel not found on system, downloading...")
    include(FetchContent)

    FetchContent_Declare(
    SortFilterProxyModel
    GIT_REPOSITORY https://github.com/valaczka/SortFilterProxyModel.git
    GIT_TAG 63cf8358fc54044197a4d40d7b7900224e008023) # 28-02-2024

    set(QT_VERSION_MAJOR
        6
        CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(SortFilterProxyModel)

    set_target_properties(SortFilterProxyModel
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
