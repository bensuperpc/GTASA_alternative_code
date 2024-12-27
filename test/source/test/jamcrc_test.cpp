#include <string>
#include "GTA_SA_cheat_finder_openmp.hpp"
#include "GTA_SA_cheat_finder_stdthread.hpp"

#if defined(BUILD_WITH_CUDA)
#include "GTA_SA_cheat_finder_cuda.hpp"
#include "cuda/wrapper.hpp"
#endif  // BUILD_WITH_CUDA

#if defined(BUILD_WITH_OPENCL)
#include "GTA_SA_cheat_finder_opencl.hpp"
#include "opencl/wrapper.hpp"
#endif  // BUILD_WITH_OPENCL

#include "gtest/gtest.h"

TEST(jamcrc, basic1) {
    std::string str = "";
    uint32_t expected_crc = 0xffffffff;

    std::reverse(str.begin(), str.end());
    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    EXPECT_EQ(expected_crc, gtaSA.jamcrc(str));
}

TEST(jamcrc, basic2) {
    std::string str = "ASNAEB";
    uint32_t expected_crc = 0x555fc201;

    std::reverse(str.begin(), str.end());
    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    EXPECT_EQ(expected_crc, gtaSA.jamcrc(str));
}

TEST(jamcrc, basic3) {
    std::string str = "ASBHGRB";
    uint32_t expected_crc = 0xa7613f99;

    std::reverse(str.begin(), str.end());
    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    EXPECT_EQ(expected_crc, gtaSA.jamcrc(str));
}

TEST(jamcrc, basic4) {
    std::string str = "XICWMD";
    uint32_t expected_crc = 0x1a9aa3d6;

    std::reverse(str.begin(), str.end());
    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    EXPECT_EQ(expected_crc, gtaSA.jamcrc(str));
}

TEST(jamcrc, basic5) {
    std::string str = "LGBTQIA+";
    uint32_t expected_crc = 0x6ba88a6;

    std::reverse(str.begin(), str.end());
    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    EXPECT_EQ(expected_crc, gtaSA.jamcrc(str));
}

TEST(jamcrc, basic6) {
    std::string str = "intergouvernementalisations";
    uint32_t expected_crc = 0x1a384955;

    std::reverse(str.begin(), str.end());
    GTA_SA_OPENMP gtaSA = GTA_SA_OPENMP();

    EXPECT_EQ(expected_crc, gtaSA.jamcrc(str));
}

#if defined(BUILD_WITH_CUDA)
TEST(jamcrc, cuda_basic1) {
    std::string str = "";
    uint32_t expected_crc = 0x0;

    std::reverse(str.begin(), str.end());

    auto const crc = my::cuda::jamcrc(str.data(), str.size(), 0, 64);

    EXPECT_EQ(expected_crc, crc);
}

TEST(jamcrc, cuda_basic2) {
    std::string str = "ASNAEB";
    uint32_t expected_crc = 0x555fc201;

    std::reverse(str.begin(), str.end());

    auto const crc = my::cuda::jamcrc(str.data(), str.size(), 0, 64);

    EXPECT_EQ(expected_crc, crc);
}

TEST(jamcrc, cuda_basic3) {
    std::string str = "ASBHGRB";
    uint32_t expected_crc = 0xa7613f99;

    std::reverse(str.begin(), str.end());

    auto const crc = my::cuda::jamcrc(str.data(), str.size(), 0, 64);

    EXPECT_EQ(expected_crc, crc);
}

TEST(jamcrc, cuda_basic4) {
    std::string str = "XICWMD";
    uint32_t expected_crc = 0x1a9aa3d6;

    std::reverse(str.begin(), str.end());

    auto const crc = my::cuda::jamcrc(str.data(), str.size(), 0, 64);

    EXPECT_EQ(expected_crc, crc);
}

TEST(jamcrc, cuda_basic5) {
    std::string str = "LGBTQIA+";
    uint32_t expected_crc = 0x6ba88a6;

    std::reverse(str.begin(), str.end());

    auto const crc = my::cuda::jamcrc(str.data(), str.size(), 0, 64);

    EXPECT_EQ(expected_crc, crc);
}

TEST(jamcrc, cuda_basic6) {
    std::string str = "intergouvernementalisations";
    uint32_t expected_crc = 0x1a384955;

    std::reverse(str.begin(), str.end());

    auto const crc = my::cuda::jamcrc(str.data(), str.size(), 0, 64);

    EXPECT_EQ(expected_crc, crc);
}

#endif

#if defined(BUILD_WITH_OPENCL)
TEST(jamcrc, opencl_basic1) {
    std::string str = "";
    uint32_t expected_crc = 0xffffffff;

    std::reverse(str.begin(), str.end());

    auto const crc = my::opencl::jamcrc(str.data(), str.size(), 0);

    EXPECT_EQ(expected_crc, crc);
}

TEST(jamcrc, opencl_basic2) {
    std::string str = "ASNAEB";
    uint32_t expected_crc = 0x555fc201;

    std::reverse(str.begin(), str.end());

    auto const crc = my::opencl::jamcrc(str.data(), str.size(), 0);

    EXPECT_EQ(expected_crc, crc);
}

TEST(jamcrc, opencl_basic3) {
    std::string str = "ASBHGRB";
    uint32_t expected_crc = 0xa7613f99;

    std::reverse(str.begin(), str.end());

    auto const crc = my::opencl::jamcrc(str.data(), str.size(), 0);

    EXPECT_EQ(expected_crc, crc);
}

TEST(jamcrc, opencl_basic4) {
    std::string str = "XICWMD";
    uint32_t expected_crc = 0x1a9aa3d6;

    std::reverse(str.begin(), str.end());

    auto const crc = my::opencl::jamcrc(str.data(), str.size(), 0);

    EXPECT_EQ(expected_crc, crc);
}

TEST(jamcrc, opencl_basic5) {
    std::string str = "LGBTQIA+";
    uint32_t expected_crc = 0x6ba88a6;

    std::reverse(str.begin(), str.end());

    auto const crc = my::opencl::jamcrc(str.data(), str.size(), 0);

    EXPECT_EQ(expected_crc, crc);
}

TEST(jamcrc, opencl_basic6) {
    std::string str = "intergouvernementalisations";
    uint32_t expected_crc = 0x1a384955;

    std::reverse(str.begin(), str.end());

    auto const crc = my::opencl::jamcrc(str.data(), str.size(), 0);

    EXPECT_EQ(expected_crc, crc);
}
#endif

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
