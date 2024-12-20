#include <array>
#include <string>

#include "GTA_SA_cheat_finder_openmp.hpp"
#include "GTA_SA_cheat_finder_stdthread.hpp"
#if defined(BUILD_WITH_CUDA)
#include "GTA_SA_cheat_finder_cuda.hpp"
#endif  // BUILD_WITH_CUDA

#include "gtest/gtest.h"

static const auto array_size = 29;

TEST(jamcrc, basic1) {
    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    std::string str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::array<char, array_size> result = {0};

    for (auto i = 1; i <= str.size(); i++) {
        gtaSA.generateString(result.data(), i);
        std::string expected = str.substr(i - 1, 1);
        EXPECT_EQ(static_cast<std::string>(result.data()), expected);
        std::cout << "Expected: " << expected << " Result: " << static_cast<std::string>(result.data()) << "For i: " << i << std::endl;

        result.fill(0);
    }
}

TEST(jamcrc, basic2) {
    std::array<char, array_size> result = {0};

    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    gtaSA.generateString(result.data(), 255);

    EXPECT_EQ(static_cast<std::string>(result.data()), "UI");
}

TEST(jamcrc, basic3) {
    std::array<char, array_size> result = {0};

    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    gtaSA.generateString(result.data(), 40000);

    EXPECT_EQ(static_cast<std::string>(result.data()), "LDGB");
}

TEST(jamcrc, basic4) {
    std::array<char, array_size> result = {0};

    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    gtaSA.generateString(result.data(), 1000000000);

    EXPECT_EQ(static_cast<std::string>(result.data()), "LXSGDFC");
}

TEST(jamcrc, basic5) {
    std::array<char, array_size> result = {0};

    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    gtaSA.generateString(result.data(), 1000000000000);

    EXPECT_EQ(static_cast<std::string>(result.data()), "NUXRHCMTD");
}

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
