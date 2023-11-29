cmake_minimum_required(VERSION 3.14.0)

include(FetchContent)

FetchContent_Declare(
  backward-cpp
  GIT_REPOSITORY https://github.com/bombela/backward-cpp.git
  GIT_TAG 65a769ffe77cf9d759d801bc792ac56af8e911a3) # 2023-10-12

FetchContent_MakeAvailable(backward-cpp)

# Add ${BACKWARD_ENABLE} to your executable and add add_backward(<executable>) to your CMakeLists.txt
include_directories(${backward-cpp_SOURCE_DIR})