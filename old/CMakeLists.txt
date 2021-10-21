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

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake)

include(ConfigureNinja)

#=== C++ VERSION CHECK ===
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_STANDARD 17)
set(CXX_EXTENSIONS OFF)


target_compile_options(${TARGET_NAME} PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow>
)

if (UNIX)
    message(STATUS "Using emscripten: disable LTO")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fno-exceptions")
endif()
if (MSVC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4")
endif()

#target_compile_options(gta PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
#set_target_properties(gta PROPERTIES CXX_STANDARD 17 CMAKE_CXX_STANDARD_REQUIRED OFF)

add_executable(gta src/GTA_SA_cheat_finder.hpp src/GTA_SA_cheat_finder.cpp)

include(ConfigureOpenMP)

# Fix long build time with emscripten
if(EMSCRIPTEN)
    message(STATUS "Using emscripten: disable LTO")
else()
    include(ConfigureLTO)
endif()
