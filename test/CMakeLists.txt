# Parent project does not export its library target, so this CML implicitly
# depends on being added from it, i.e. the testing is done only from the build
# tree and is not feasible from an install location

project(GTA_SA_cheat_finderTests LANGUAGES CXX)

add_executable(GTA_SA_cheat_finder_test source/GTA_SA_cheat_finder_test.cpp)
target_link_libraries(GTA_SA_cheat_finder_test PRIVATE GTA_SA_cheat_finder_lib)
target_compile_features(GTA_SA_cheat_finder_test PRIVATE cxx_std_17)

add_test(NAME GTA_SA_cheat_finder_test COMMAND GTA_SA_cheat_finder_test)
