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
    opencl_lib
    VERSION 0.1.0
    DESCRIPTION "Krack cuda lib"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES C CXX
)

find_package(OpenCL 2.0 REQUIRED)

set(SRCS
    wrapper.cpp
    kernel.cpp
	)
	
set(HEADERS
    wrapper.hpp
    kernel.hpp
    )

add_library(opencl_lib ${SRCS} ${HEADERS})

#target_include_directories(opencl_lib PUBLIC ${OpenCL_INCLUDE_DIRS})
#target_link_libraries(opencl_lib PUBLIC ${OpenCL_LIBRARIES})

#target_include_directories(opencl_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(opencl_lib PRIVATE OpenCL::OpenCL)
#target_compile_features(opencl_lib PUBLIC cuda_std_17)

set_target_properties(opencl_lib
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
