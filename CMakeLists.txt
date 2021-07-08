##############################################################
#   ____                                                     #
#  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___    #
#  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __|   #
#  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__    #
#  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___|   #
#                             |_|             |_|            #
##############################################################
#                                                            #
#  BenLib, 2020                                              #
#  Created: 05, February, 2020                               #
#  Modified: 06, June, 2021                                  #
#  file: CMakeLists.txt                                      #
#  CMake                                                     #
#  Source:                                                   #
#                                                            #
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################

project(gta LANGUAGES CXX)
cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CXX_EXTENSIONS OFF)


find_package(OpenMP)

add_executable(gta src/GTA_SA_cheat_finder.hpp src/GTA_SA_cheat_finder.cpp)

target_compile_options(gta PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
set_target_properties(gta PROPERTIES CXX_STANDARD 17)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gta PUBLIC OpenMP::OpenMP_CXX)
endif()


#   ================    LTO    ================
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

if( supported )
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET gta PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()
