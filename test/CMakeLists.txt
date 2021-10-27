# Parent project does not export its library target, so this CML implicitly
# depends on being added from it, i.e. the testing is done only from the build
# tree and is not feasible from an install location

project(GTA_SA_cheat_finderTests LANGUAGES CXX)

add_executable(find_string_inv source/find_string_inv.cpp)
target_link_libraries(find_string_inv PRIVATE GTA_SA_cheat_finder_lib)
target_compile_features(find_string_inv PRIVATE cxx_std_17)

add_test(NAME find_string_inv COMMAND find_string_inv)

add_executable(jamcrc source/jamcrc.cpp)
target_link_libraries(jamcrc PRIVATE GTA_SA_cheat_finder_lib)
target_compile_features(jamcrc PRIVATE cxx_std_17)

add_test(NAME jamcrc COMMAND jamcrc)

add_executable(precompute_crc source/precompute_crc.cpp)
target_link_libraries(precompute_crc PRIVATE GTA_SA_cheat_finder_lib)
target_compile_features(precompute_crc PRIVATE cxx_std_17)

add_test(NAME precompute_crc COMMAND precompute_crc)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        ec2d5013fbeddd30e65c44639287b15eb591d68e
)

FetchContent_MakeAvailable(Catch2)

add_executable(find_string_inv_catch source/find_string_inv_catch.cpp)
target_link_libraries(find_string_inv_catch PRIVATE Catch2::Catch2WithMain)
add_test(NAME find_string_inv_catch COMMAND find_string_inv_catch)