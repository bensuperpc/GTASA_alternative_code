##############################################################
#   ____                                                     #
#  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___    #
#  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __|   #
#  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__    #
#  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___|   #
#                             |_|             |_|            #
##############################################################
#                                                            #
#  BenLib, 2021                                              #
#  Created: 16, March, 2021                                  #
#  Modified: 17, March, 2021                                 #
#  file: CMakeLists.txt                                      #
#  CMake                                                     #
#  Source:                                                   #
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################

cmake_minimum_required(VERSION 3.14)
project(
    GTA_SA_cheat_finder_lib
    VERSION 0.1.0
    DESCRIPTION "GTA_SA_cheat_finder lib"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES CXX
)

find_package(OpenMP 2.0)

find_package(CUDAToolkit 11.0)

find_package(OpenCL 2.0)

# Download ThreadPool
include(../../cmake/lib/threadpool.cmake)

if (CUDAToolkit_FOUND)
  add_subdirectory(cuda)
endif()

if (OpenCL_FOUND)
  add_subdirectory(opencl)
endif()

set(SRCS
    GTA_SA_cheat_finder_main.cpp
    state/GTA_SA_cheat_finder_stdthread.cpp
    state/GTA_SA_cheat_finder_virtual.cpp
    GTA_SA_cheat_finder_result.cpp
	)
	
set(HEADERS
    GTA_SA_cheat_finder_main.hpp
    state/GTA_SA_cheat_finder_stdthread.hpp
    state/GTA_SA_cheat_finder_virtual.hpp
    GTA_SA_cheat_finder_result.hpp
    )

if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  list(APPEND SRCS state/GTA_SA_cheat_finder_openmp.cpp)
  list(APPEND HEADERS state/GTA_SA_cheat_finder_openmp.hpp)
endif()

if (CUDAToolkit_FOUND)
  list(APPEND SRCS state/GTA_SA_cheat_finder_cuda.cpp)
  list(APPEND HEADERS state/GTA_SA_cheat_finder_cuda.hpp)
endif()

if (OpenCL_FOUND)
  list(APPEND SRCS state/GTA_SA_cheat_finder_opencl.cpp)
  list(APPEND HEADERS state/GTA_SA_cheat_finder_opencl.hpp)
endif()
  
add_library(GTA_SA_cheat_finder_lib ${SRCS} ${HEADERS})

target_compile_features(GTA_SA_cheat_finder_lib PUBLIC cxx_std_20)

target_include_directories(
    GTA_SA_cheat_finder_lib ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/state>"
)

if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
    target_link_libraries(GTA_SA_cheat_finder_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

if (CUDAToolkit_FOUND)
  target_include_directories(GTA_SA_cheat_finder_lib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(GTA_SA_cheat_finder_lib PUBLIC CUDA::cudart)
  target_link_libraries(GTA_SA_cheat_finder_lib PUBLIC cuda_lib)
endif()

if (OpenCL_FOUND)
  target_include_directories(GTA_SA_cheat_finder_lib PUBLIC ${OpenCL_INCLUDE_DIRS})
  target_link_libraries(GTA_SA_cheat_finder_lib PUBLIC ${OpenCL_LIBRARIES})
  target_link_libraries(GTA_SA_cheat_finder_lib PUBLIC opencl_lib)
endif()

set_target_properties(GTA_SA_cheat_finder_lib
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

