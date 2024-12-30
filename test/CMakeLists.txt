cmake_minimum_required(VERSION 3.14)

project(GTASA_alternative_code_tests LANGUAGES C CXX)

if (NOT Threads_FOUND)
    find_package(Threads REQUIRED)
endif()

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

include(../cmake/project-is-top-level.cmake)
include(../cmake/folders.cmake)
include(../cmake/windows-set-path.cmake)

function(test_bench_generator TEST_BENCH_NAME IS_TEST ADD_TO_TEST)
    if (IS_TEST)
        add_executable("${TEST_BENCH_NAME}" "source/test/${TEST_BENCH_NAME}.cpp")
    else()
        add_executable("${TEST_BENCH_NAME}" "source/benchmark/${TEST_BENCH_NAME}.cpp")
    endif()

    if (IS_TEST)
        target_link_libraries("${TEST_BENCH_NAME}" PRIVATE gtest)
    else()
        target_link_libraries("${TEST_BENCH_NAME}" PRIVATE benchmark::benchmark)
    endif()

    target_link_libraries("${TEST_BENCH_NAME}" PRIVATE GTASA_alternative_code_lib)

    if (ENABLE_OPENMP)
        target_link_libraries("${TEST_BENCH_NAME}" PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
    endif()
  
    if (ENABLE_CUDA)
        target_link_libraries("${TEST_BENCH_NAME}" PRIVATE cuda_lib)
    endif()

    if (ENABLE_OPENCL)
        target_link_libraries("${TEST_BENCH_NAME}" PRIVATE opencl_lib)
    endif()

    set_target_properties("${TEST_BENCH_NAME}"
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    #target_compile_features("${TEST_BENCH_NAME}" PRIVATE cxx_std_17)
    target_compile_definitions(GTASA_alternative_code_lib PUBLIC ENABLE_CUDA=$<IF:$<BOOL:${ENABLE_CUDA}>,1,0>)
    target_compile_definitions(GTASA_alternative_code_lib PUBLIC ENABLE_OPENCL=$<IF:$<BOOL:${ENABLE_OPENCL}>,1,0>)
    target_compile_definitions(GTASA_alternative_code_lib PUBLIC ENABLE_OPENMP=$<IF:$<BOOL:${ENABLE_OPENMP}>,1,0>)

    #if(NOT CMAKE_BUILD_TYPE MATCHES Debug AND NOT CMAKE_BUILD_TYPE MATCHES Coverage)
    #  add_test(NAME "${TEST_BENCH_NAME}" COMMAND $<TARGET_FILE:${TEST_BENCH_NAME}>)
    #elseif()
    #  message(STATUS "Disable ${BENCH_NAME}, Performance benchmark test only run on Release/RelWithDebInfo/MinSizeRel")
    #endif()

    if (ADD_TO_TEST)
        add_test(NAME "${TEST_BENCH_NAME}" COMMAND $<TARGET_FILE:${TEST_BENCH_NAME}>)
    endif()
    
endfunction()

if(NOT WIN32)
    add_subdirectory(../external/googletest external/googletest)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_FORCE_WERROR OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_CXX_LINKER_FLAGS "" CACHE STRING "" FORCE)
    set(BENCHMARK_CXX_LIBRARIES "" CACHE STRING "" FORCE)
    set(BENCHMARK_CXX_FLAGS "" CACHE STRING "" FORCE)
    set(BENCHMARK_PRIVATE_LINK_LIBRARIES "" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_COVERAGE "" CACHE STRING "" FORCE)
    set(CMAKE_REQUIRED_FLAGS "" CACHE STRING "" FORCE)
    add_subdirectory(../external/benchmark external/benchmark)

    # Add tests
    #test_bench_generator(find_string_inv_test true true)
    #test_bench_generator(jamcrc_test true true)
    
    test_bench_generator(gta_sa_test_threadpool true true)
    if(ENABLE_OPENMP)
        test_bench_generator(gta_sa_test_openmp true true)
    endif()
    if(ENABLE_CUDA)
        test_bench_generator(gta_sa_test_cuda true true)
    endif()
    
    # Add bench
    #test_bench_generator(find_string_inv_bench false false)
    #test_bench_generator(jamcrc_bench false false)
  
endif()

# ---- End-of-file commands ----

add_folders(Test)
