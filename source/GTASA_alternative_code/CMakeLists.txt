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
    GTASA_alternative_code_lib
    VERSION 0.1.0
    DESCRIPTION "GTASA_alternative_code lib"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES CXX
)

find_package(OpenMP 2.0)

find_package(CUDAToolkit 11.0)

#find_package(OpenCL 2.0)

if (CUDAToolkit_FOUND)
  add_subdirectory(module/cuda/kernel)
endif()

if (OpenCL_FOUND)
  #add_subdirectory(opencl)
endif()

set(SRCS
    GTASAResult.cpp
    GTASARequest.cpp
    GTASAEngine.cpp
    module/GTASAModule.cpp
    module/GTASAModuleThreadpool.cpp
    #module/GTASAModuleOpenCL.cpp
	)
	
set(HEADERS
    GTASAResult.hpp
    GTASARequest.hpp
    GTASAEngine.hpp
    module/GTASAModule.hpp
    module/GTASAModuleThreadpool.hpp
    #module/GTASAModuleOpenCL.hpp
    )

if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  list(APPEND SRCS module/GTASAModuleOpenMP.cpp)
  list(APPEND HEADERS module/GTASAModuleOpenMP.hpp)
endif()

if (CUDAToolkit_FOUND)
  list(APPEND SRCS module/cuda/GTASAModuleCUDA.cpp)
  list(APPEND HEADERS  module/cuda/GTASAModuleCUDA.hpp)
endif()

if (OpenCL_FOUND)
  #list(APPEND SRCS state/GTASA_alternative_code_opencl.cpp)
  #list(APPEND HEADERS state/GTASA_alternative_code_opencl.hpp)
endif()
  
add_library(GTASA_alternative_code_lib ${SRCS} ${HEADERS})

target_include_directories(
    GTASA_alternative_code_lib ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/state>"
)

if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
    target_link_libraries(GTASA_alternative_code_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

if (CUDAToolkit_FOUND)
  target_include_directories(GTASA_alternative_code_lib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(GTASA_alternative_code_lib PUBLIC CUDA::cudart)
  target_link_libraries(GTASA_alternative_code_lib PUBLIC cuda_lib)
endif()

if (OpenCL_FOUND)
  #target_include_directories(GTASA_alternative_code_lib PUBLIC ${OpenCL_INCLUDE_DIRS})
  #target_link_libraries(GTASA_alternative_code_lib PUBLIC ${OpenCL_LIBRARIES})
  #target_link_libraries(GTASA_alternative_code_lib PUBLIC opencl_lib)
endif()

set_target_properties(GTASA_alternative_code_lib
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

