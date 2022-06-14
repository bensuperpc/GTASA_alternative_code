cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    GTA_SA_cheat_finder
    VERSION 2.0.0
    DESCRIPTION "Find alternative cheat code in Grand Theft Auto San Andreas (2004) by bruteforce"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES C CXX
)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
endif()

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Enable Utile ----
include(cmake/utile/ccache.cmake)
# include(cmake/utile/lto.cmake)
# include(cmake/utile/ninja_color.cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- Declare deps ----

find_package(OpenMP 2.0)

find_package(CUDAToolkit 11.0)

#find_package(OpenCL 2.0)

# Download ThreadPool
include(FetchContent)
FetchContent_Declare(
    thread-pool
    FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/_deps
    URL      https://github.com/bshoshany/thread-pool/archive/refs/tags/v3.0.0.zip
    URL_HASH SHA512=bab8a512467d6c9b964bcab11f4c48b91ce0bc68874d368fa1975a1d8036f9891d47091c6e0a3096c7b5bcb8455997184a438e4f6936a8f8df4d22df8fa8fba6
)
FetchContent_MakeAvailable(thread-pool)

if (CUDAToolkit_FOUND)
  add_subdirectory(source/cuda)
endif()

# ---- Declare library ----

add_library(
    GTA_SA_cheat_finder_lib OBJECT
    source/GTA_SA_cheat_finder.cpp
)

target_include_directories(
    GTA_SA_cheat_finder_lib ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_include_directories(
    GTA_SA_cheat_finder_lib ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${thread-pool_SOURCE_DIR}>"
)

target_compile_features(GTA_SA_cheat_finder_lib PUBLIC cxx_std_20)

if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  target_link_libraries(GTA_SA_cheat_finder_lib PUBLIC OpenMP::OpenMP_CXX)
endif()


if (CUDAToolkit_FOUND)
  target_include_directories(GTA_SA_cheat_finder_lib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(GTA_SA_cheat_finder_lib PUBLIC CUDA::cudart)
  target_link_libraries(GTA_SA_cheat_finder_lib PUBLIC cuda_lib)

  # target_include_directories(GTA_SA_cheat_finder_lib PUBLIC "${PROJECT_SOURCE_DIR}/source/cuda")
endif()

# ---- Declare executable ----

add_executable(GTA_SA_cheat_finder_exe source/main.cpp)
add_executable(GTA_SA_cheat_finder::exe ALIAS GTA_SA_cheat_finder_exe)

set_target_properties(
    GTA_SA_cheat_finder_exe PROPERTIES
    OUTPUT_NAME GTA_SA_cheat_finder
    EXPORT_NAME exe
)

target_compile_features(GTA_SA_cheat_finder_exe PRIVATE cxx_std_20)

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
