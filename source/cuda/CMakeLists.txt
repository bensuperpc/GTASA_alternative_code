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
    cuda_lib
    VERSION 0.1.0
    DESCRIPTION "Krack cuda lib"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES C CXX
)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
endif()

find_package(CUDAToolkit 11.0)

set(SRCS
    kernel.cu
    wrapper.cu
	)
	
set(HEADERS
    kernel.cuh
    wrapper.hpp
    )

add_library(cuda_lib ${SRCS} ${HEADERS})

#if (CUDAToolkit_FOUND)
target_include_directories(cuda_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cuda_lib PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cuda_lib PRIVATE CUDA::cudart)
target_compile_features(cuda_lib PUBLIC cuda_std_17)
#endif()

set_target_properties(cuda_lib
PROPERTIES
RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
