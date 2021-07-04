project(gta LANGUAGES CXX)
cmake_minimum_required(VERSION 3.9)

find_package(OpenMP)

add_executable(gta src/GTA_SA_cheat_finder.hpp src/GTA_SA_cheat_finder.cpp)

target_compile_options(gta PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
set_target_properties(gta PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gta PUBLIC OpenMP::OpenMP_CXX)
endif()
