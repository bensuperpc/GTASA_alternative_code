# Parent project does not export its library target, so this CML implicitly
# depends on being added from it, i.e. the testing is done only from the build
# tree and is not feasible from an install location

project(GTA_SA_cheat_finderTests LANGUAGES CXX)

add_executable(find_string_inv source/find_string_inv.cpp)
target_link_libraries(find_string_inv PRIVATE GTA_SA_cheat_finder_lib)
# target_compile_features(find_string_inv PRIVATE cxx_std_17)

add_test(NAME find_string_inv COMMAND find_string_inv)

add_executable(jamcrc source/jamcrc.cpp)
target_link_libraries(jamcrc PRIVATE GTA_SA_cheat_finder_lib)
# target_compile_features(jamcrc PRIVATE cxx_std_17)

add_test(NAME jamcrc COMMAND jamcrc)

add_executable(precompute_crc source/precompute_crc.cpp)
target_link_libraries(precompute_crc PRIVATE GTA_SA_cheat_finder_lib)
# target_compile_features(precompute_crc PRIVATE cxx_std_17)

add_test(NAME precompute_crc COMMAND precompute_crc)

if(UNIX AND NOT APPLE)
  include(FetchContent)

  # Externally provided libraries
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0)

  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/bensuperpc/benchmark.git
    GIT_TAG main-alt)

  FetchContent_MakeAvailable(googletest googlebenchmark)

  add_executable(jamcrc_bench source/jamcrc_bench.cpp)
  target_link_libraries(jamcrc_bench PRIVATE benchmark::benchmark
                                             GTA_SA_cheat_finder_lib)
  add_test(NAME jamcrc_bench COMMAND jamcrc_bench)

  add_executable(find_string_inv_bench source/find_string_inv_bench.cpp)
  target_link_libraries(find_string_inv_bench PRIVATE benchmark::benchmark
                                                      GTA_SA_cheat_finder_lib)
  add_test(NAME find_string_inv_bench COMMAND find_string_inv_bench)
endif(UNIX)