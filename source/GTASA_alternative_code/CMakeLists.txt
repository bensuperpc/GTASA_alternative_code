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
    LANGUAGES C CXX
)

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

if (ENABLE_CUDA)
    add_subdirectory(module/cuda/kernel)
endif()

if (ENABLE_OPENCL)
  #add_subdirectory(opencl)
endif()

if (ENABLE_THREADPOOL)
    include_directories(../../external/thread-pool/include)
endif()

set(SRCS
    GTASAResult.cpp
    GTASARequest.cpp
    GTASAEngine.cpp
    module/GTASAModule.cpp
    module/mono/GTASAModuleMono.cpp
	)
	
set(HEADERS
    GTASAResult.hpp
    GTASARequest.hpp
    GTASAEngine.hpp
    module/GTASAModule.hpp
    module/mono/GTASAModuleMono.hpp
    )

if (ENABLE_THREADPOOL)
    list(APPEND SRCS module/threadpool/GTASAModuleThreadpool.cpp)
    list(APPEND HEADERS module/threadpool/GTASAModuleThreadpool.hpp)
endif()

if (ENABLE_OPENMP)
  list(APPEND SRCS module/openmp/GTASAModuleOpenMP.cpp)
  list(APPEND HEADERS module/openmp/GTASAModuleOpenMP.hpp)
endif()

if (ENABLE_CUDA)
  list(APPEND SRCS module/cuda/GTASAModuleCUDA.cpp)
  list(APPEND HEADERS  module/cuda/GTASAModuleCUDA.hpp)
endif()

if (ENABLE_OPENCL)
  #list(APPEND SRCS module/opencl/GTASAModuleOpenCL.cpp)
  #list(APPEND HEADERS module/opencl/GTASAModuleOpenCL.hpp)
endif()
  
add_library(GTASA_alternative_code_lib ${SRCS} ${HEADERS})

target_include_directories(
    GTASA_alternative_code_lib ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/state>"
)

if(ENABLE_THREADPOOL)
    target_link_libraries(GTASA_alternative_code_lib PRIVATE Threads::Threads)
endif()

if (ENABLE_OPENMP)
    target_link_libraries(GTASA_alternative_code_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

if (ENABLE_CUDA)
    target_include_directories(GTASA_alternative_code_lib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(GTASA_alternative_code_lib PUBLIC CUDA::cudart)
    target_link_libraries(GTASA_alternative_code_lib PUBLIC cuda_lib)
endif()

if (ENABLE_OPENCL)
  target_include_directories(GTASA_alternative_code_lib PUBLIC ${OpenCL_INCLUDE_DIRS})
  target_link_libraries(GTASA_alternative_code_lib PUBLIC ${OpenCL_LIBRARIES})
  target_link_libraries(GTASA_alternative_code_lib PUBLIC opencl_lib)
endif()

target_compile_definitions(GTASA_alternative_code_lib PUBLIC ENABLE_CUDA=$<IF:$<BOOL:${ENABLE_CUDA}>,1,0>)
target_compile_definitions(GTASA_alternative_code_lib PUBLIC ENABLE_OPENCL=$<IF:$<BOOL:${ENABLE_OPENCL}>,1,0>)
target_compile_definitions(GTASA_alternative_code_lib PUBLIC ENABLE_OPENMP=$<IF:$<BOOL:${ENABLE_OPENMP}>,1,0>)
target_compile_definitions(GTASA_alternative_code_lib PUBLIC ENABLE_THREADPOOL=$<IF:$<BOOL:${ENABLE_THREADPOOL}>,1,0>)

set_target_properties(GTASA_alternative_code_lib
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

