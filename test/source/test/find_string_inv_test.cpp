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
    std::array<char, array_size> tmp1 = {0};

    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    gtaSA.find_string_inv(tmp1.data(), 0);

    EXPECT_NE("", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("B", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("C", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("AA", static_cast<std::string>(tmp1.data()));

    EXPECT_EQ("A", static_cast<std::string>(tmp1.data()));
}

TEST(jamcrc, basic2) {
    std::array<char, array_size> tmp1 = {0};

    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    gtaSA.find_string_inv(tmp1.data(), 255);

    EXPECT_NE("", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("A", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("B", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("C", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("AA", static_cast<std::string>(tmp1.data()));

    EXPECT_EQ("UI", static_cast<std::string>(tmp1.data()));
}

TEST(jamcrc, basic3) {
    std::array<char, array_size> tmp1 = {0};

    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    gtaSA.find_string_inv(tmp1.data(), 40000);

    EXPECT_NE("", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("A", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("B", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("C", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("AA", static_cast<std::string>(tmp1.data()));

    EXPECT_EQ("LDGB", static_cast<std::string>(tmp1.data()));
}

TEST(jamcrc, basic4) {
    std::array<char, array_size> tmp1 = {0};

    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    gtaSA.find_string_inv(tmp1.data(), 1000000000);

    EXPECT_NE("", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("A", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("B", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("C", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("AA", static_cast<std::string>(tmp1.data()));

    EXPECT_EQ("LXSGDFC", static_cast<std::string>(tmp1.data()));
}

TEST(jamcrc, basic5) {
    std::array<char, array_size> tmp1 = {0};

    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    gtaSA.find_string_inv(tmp1.data(), 1000000000000);

    EXPECT_NE("", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("A", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("B", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("C", static_cast<std::string>(tmp1.data()));
    EXPECT_NE("AA", static_cast<std::string>(tmp1.data()));

    EXPECT_EQ("NUXRHCMTD", static_cast<std::string>(tmp1.data()));
}

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
