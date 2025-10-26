cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    GTASA_alternative_code
    VERSION 2.0.0
    DESCRIPTION "Find alternative cheat code in Grand Theft Auto San Andreas (2004) by bruteforce"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES C CXX
)

# ---- Enable Utile ----
include(cmake/utile/ccache.cmake)
#include(cmake/utile/sccache.cmake)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- CUDA ----
if (NOT DEFINED ENABLE_CUDA)
    find_package(CUDAToolkit 11.0)
    set(ENABLE_CUDA ${CUDAToolkit_FOUND} CACHE BOOL "Enable CUDA" FORCE)
elseif (ENABLE_CUDA AND (NOT DEFINED CUDAToolkit_FOUND OR NOT CUDAToolkit_FOUND))
    find_package(CUDAToolkit 11.0 REQUIRED)
endif()

if (ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    endif()
endif()

# ---- OpenCL ----
if (NOT DEFINED ENABLE_OPENCL)
    find_package(OpenCL 2.0)
    set(ENABLE_OPENCL ${OpenCL_FOUND} CACHE BOOL "Enable OpenCL" FORCE)
elseif (ENABLE_OPENCL AND (NOT DEFINED OpenCL_FOUND OR NOT OpenCL_FOUND))
    find_package(OpenCL 2.0 REQUIRED)
endif()

# ---- OpenMP ----
if (NOT DEFINED ENABLE_OPENMP)
    find_package(OpenMP 2.0)
    set(ENABLE_OPENMP ${OpenMP_FOUND} CACHE BOOL "Enable OpenMP" FORCE)
elseif (ENABLE_OPENMP AND (NOT DEFINED OpenMP_FOUND OR NOT OpenMP_FOUND))
    find_package(OpenMP 2.0 REQUIRED)
endif()

# ---- Threadpool ----
if (NOT DEFINED ENABLE_THREADPOOL)
    find_package(Threads)
    set(ENABLE_THREADPOOL ${Threads_FOUND} CACHE BOOL "Enable ThreadPool" FORCE)
elseif (ENABLE_THREADPOOL AND (NOT DEFINED Threads_FOUND OR NOT Threads_FOUND))
    find_package(Threads REQUIRED)
endif()

# TODO: Change to target_include_directories
add_subdirectory(source/GTASA_alternative_code)

# ---- Declare executable ----

set(SRCS
    source/main.cpp
	)
	
set(HEADERS
    )

add_executable(GTASA_alternative_code_exe
    ${SRCS} ${HEADERS}
)

add_executable(GTASA_alternative_code::exe ALIAS GTASA_alternative_code_exe)

set_target_properties(GTASA_alternative_code_exe PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

if(ENABLE_THREADPOOL)
    target_link_libraries(GTASA_alternative_code_exe PRIVATE Threads::Threads)
endif()

if (ENABLE_OPENMP)
  target_link_libraries(GTASA_alternative_code_exe PUBLIC OpenMP::OpenMP_CXX)
endif()

if (ENABLE_CUDA)
  target_include_directories(GTASA_alternative_code_exe PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(GTASA_alternative_code_exe PUBLIC GTASA_alternative_code_lib)

set_property(TARGET GTASA_alternative_code_exe PROPERTY OUTPUT_NAME GTASA_alternative_code)

target_compile_definitions(GTASA_alternative_code_exe PUBLIC PROJECT_VERSION="${PROJECT_VERSION}")
target_compile_definitions(GTASA_alternative_code_exe PUBLIC CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
target_compile_definitions(GTASA_alternative_code_exe PUBLIC CMAKE_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}")
target_compile_definitions(GTASA_alternative_code_exe PUBLIC CMAKE_C_COMPILER_ID="${CMAKE_C_COMPILER_ID}")
target_compile_definitions(GTASA_alternative_code_exe PUBLIC CMAKE_VERSION="${CMAKE_VERSION}")
target_compile_definitions(GTASA_alternative_code_exe PUBLIC ENABLE_CUDA=$<IF:$<BOOL:${ENABLE_CUDA}>,1,0>)
target_compile_definitions(GTASA_alternative_code_exe PUBLIC ENABLE_OPENCL=$<IF:$<BOOL:${ENABLE_OPENCL}>,1,0>)
target_compile_definitions(GTASA_alternative_code_exe PUBLIC ENABLE_OPENMP=$<IF:$<BOOL:${ENABLE_OPENMP}>,1,0>)
target_compile_definitions(GTASA_alternative_code_exe PUBLIC ENABLE_QT=$<IF:$<BOOL:${ENABLE_QT}>,1,0>)
target_compile_definitions(GTASA_alternative_code_exe PUBLIC ENABLE_THREADPOOL=$<IF:$<BOOL:${ENABLE_THREADPOOL}>,1,0>)

#target_compile_definitions(GTASA_alternative_code_exe PUBLIC QT_MESSAGELOGCONTEXT="1")

configure_file(image/cat_sticking_out_its_tongue.jpg ${CMAKE_BINARY_DIR}/image/cat_sticking_out_its_tongue.jpg COPYONLY)


# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT GTASA_alternative_code_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of GTASA_alternative_code"
  )
endif()

include(cmake/dev-mode.cmake)
