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

project(gta LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.9)



#=== C++ VERSION CHECK ===
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_STANDARD)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++20" COMPILER_SUPPORTS_CXX20)
    CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

    if(COMPILER_SUPPORTS_CXX20)
        set(CMAKE_CXX_STANDARD 20)
        message(STATUS "C++20: OK")
    elseif(COMPILER_SUPPORTS_CXX17)
        set(CMAKE_CXX_STANDARD 17)
        message(STATUS "C++17: OK")
    elseif(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_STANDARD 14)
        message(STATUS "C++14: OK")
        message(WARNING "C++14 is old, please use newer compiler.")
    elseif(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_STANDARD 11)
        message(STATUS "C++11: OK")
        message(WARNING "C++11 is old, please use newer compiler.")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 or above support. Please use a different C++ compiler.")
    endif()
endif()

set(CMAKE_CXX_FLAGS                "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wconversion -Wshadow")

#target_compile_options(gta PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
#set_target_properties(gta PROPERTIES CXX_STANDARD 17 CMAKE_CXX_STANDARD_REQUIRED OFF)

add_executable(gta src/GTA_SA_cheat_finder.hxx src/GTA_SA_cheat_finder.cxx)
add_executable(gtaC src/GTA_SA_cheat_finder.h src/GTA_SA_cheat_finder.c)

find_package(OpenMP)
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
    message(WARNING "IPO / LTO not supported: <${error}>")
endif()
