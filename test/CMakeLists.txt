# Parent project does not export its library target, so this CML implicitly
# depends on being added from it, i.e. the testing is done only from the build
# tree and is not feasible from an install location

project(GTA_SA_cheat_finderTests LANGUAGES CXX)

if(NOT WIN32)
  include(FetchContent)

  # Externally provided libraries
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main)

  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG main)

  # Disable tests on gtest
  set(gtest_build_tests
      OFF
      CACHE BOOL "" FORCE)
  set(gtest_build_samples
      OFF
      CACHE BOOL "" FORCE)

  # Disable tests on google benchmark
  set(BENCHMARK_ENABLE_TESTING
      OFF
      CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR
      OFF
      CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(googletest googlebenchmark)
  # Lib: gtest_main benchmark::benchmark benchmark::benchmark_main
  # target_compile_features(crc32_test PRIVATE cxx_std_17)

  add_executable(jamcrc_bench source/jamcrc_bench.cpp)
  target_link_libraries(jamcrc_bench PRIVATE benchmark::benchmark_main
                                             GTA_SA_cheat_finder_lib)

  add_executable(find_string_inv_bench source/find_string_inv_bench.cpp)
  target_link_libraries(find_string_inv_bench PRIVATE benchmark::benchmark_main
                                                      GTA_SA_cheat_finder_lib)

  if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
    add_test(NAME find_string_inv_bench COMMAND find_string_inv_bench)
    add_test(NAME jamcrc_bench COMMAND jamcrc_bench)
  elseif()
    message(STATUS "Disable crc32_bench, Performance benchmark test only run on Release/RelWithDebInfo/MinSizeRel")
  endif()

  add_executable(jamcrc_test source/jamcrc_test.cpp)
  target_link_libraries(jamcrc_test PRIVATE GTA_SA_cheat_finder_lib gtest_main)
  add_test(NAME jamcrc_test COMMAND jamcrc_test)

  add_executable(find_string_inv_test source/find_string_inv_test.cpp)
  target_link_libraries(find_string_inv_test PRIVATE GTA_SA_cheat_finder_lib gtest_main)
  add_test(NAME find_string_inv_test COMMAND find_string_inv_test)

  #include(GoogleTest)
  #gtest_discover_tests(jamcrc_test)
else()
  message(WARNING "Disable tests, only supported on Linux")
endif()
