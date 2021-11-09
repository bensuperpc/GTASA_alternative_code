cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    GTA_SA_cheat_finder
    VERSION 1.3.6
    DESCRIPTION "Find alternative cheat code in Grand Theft Auto San Andreas (2004) by bruteforce"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES CXX
)

docs_early_return()

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Output bin/lib etc... ----

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- Declare library ----

add_library(
    GTA_SA_cheat_finder_lib OBJECT
    source/gta_sa_lib.cpp
)

target_include_directories(
    GTA_SA_cheat_finder_lib ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

#target_compile_features(GTA_SA_cheat_finder_lib PUBLIC cxx_std_17)

# ---- Declare executable ----

add_executable(GTA_SA_cheat_finder_exe source/main.cpp)
add_executable(GTA_SA_cheat_finder::exe ALIAS GTA_SA_cheat_finder_exe)

set_target_properties(
    GTA_SA_cheat_finder_exe PROPERTIES
    OUTPUT_NAME GTA_SA_cheat_finder
    EXPORT_NAME exe
)

#target_compile_features(GTA_SA_cheat_finder_exe PRIVATE cxx_std_17)

find_package(OpenMP 4.0)
if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  target_link_libraries(GTA_SA_cheat_finder_exe PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(GTA_SA_cheat_finder_exe PRIVATE GTA_SA_cheat_finder_lib)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT GTA_SA_cheat_finder_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of GTA_SA_cheat_finder"
  )
endif()

include(cmake/dev-mode.cmake)
