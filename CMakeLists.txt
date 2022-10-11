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

# For Qt
set(CMAKE_AUTOMOC ON)

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

find_package(Qt6 6.2 COMPONENTS Quick Core Qml Charts Multimedia)

find_package(OpenMP 2.0)

find_package(CUDAToolkit 11.0)

#find_package(OpenCL 2.0)

# Download ThreadPool
include(FetchContent)
FetchContent_Declare(
    thread-pool
    FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/_deps
    URL      https://github.com/bshoshany/thread-pool/archive/refs/tags/v3.3.0.zip
    URL_HASH SHA512=6502461753b79d4834207106b1c477addc821b637fcdd1d39e669238637ff16fe6b5921f893f4c2ddcef3446f6880863e1b9ccbe3985bac4bd7bc2305f41ac70
)
FetchContent_MakeAvailable(thread-pool)

if (CUDAToolkit_FOUND)
  add_subdirectory(source/cuda)
endif()

# ---- Declare library ----

add_library(
    GTA_SA_cheat_finder_lib OBJECT
    source/GTA_SA_cheat_finder.cpp
    source/gta_sa_ui.cpp
    source/about_compilation.cpp
    source/compilation.cpp
    source/TableModel.cpp
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

if (Qt6_FOUND)
target_link_libraries(GTA_SA_cheat_finder_lib PUBLIC Qt6::Quick Qt6::Core Qt6::Qml Qt6::Charts Qt6::Multimedia)

qt_add_qml_module(krackx_exe
    URI qml_files
    VERSION 1.0
    RESOURCE_PREFIX /bensuperpc.com
    QML_FILES 
    source/qml/main.qml 
    source/qml/AboutPage.qml 
    source/qml/KrackPasswordPage.qml 
    source/qml/mainPage.qml 
    source/qml/SettingsPage.qml 
    source/qml/GTA_SA.qml
#    RESOURCES
#    img/Profile_400x400.jpg
)
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

if (Qt6_FOUND)
target_link_libraries(GTA_SA_cheat_finder_exe PUBLIC Qt6::Quick Qt6::Core Qt6::Qml Qt6::Charts Qt6::Multimedia)
endif()

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
