#include <string>
#include "GTA_SA_cheat_finder_openmp.hpp"
#include "GTA_SA_cheat_finder_stdthread.hpp"
#if defined(BUILD_WITH_CUDA)
#include "GTA_SA_cheat_finder_cuda.hpp"
#endif  // BUILD_WITH_CUDA

#include "gtest/gtest.h"

/*
20810792         ASNAEB         0x555fc201
75396850         FHYSTV         0x44b34866
147491485        LJSPQK         0xfeda77f7
181355281        OFVIAC         0x6c0fa650
181961057        OHDUDE         0xe958788a
198489210        PRIEBJ         0xf2aa0c1d
241414872        THGLOJ         0xcaec94ee
289334426        XICWMD         0x1a9aa3d6
299376767        YECGAA         0x40cf761
311365503        ZEIIVG         0x74d4fcb1
370535590        AEDUWNV        0x9a629401
380229391        AEZAKMI        0xe1b33eb9
535721682        ASBHGRB        0xa7613f99
*/

TEST(GTA_SA_STDTHREAD, basic_calc_base_1) {
    GTA_SA_STDTHREAD gtaSA;
    gtaSA.minRange = 0;
    gtaSA.maxRange = 60000;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 0);
}

TEST(GTA_SA_STDTHREAD, basic_calc_base_2) {
    GTA_SA_STDTHREAD gtaSA;
    gtaSA.minRange = 20810700;
    gtaSA.maxRange = 20810800;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 1);
    EXPECT_EQ(gtaSA.results[0].index, 20810792);
    EXPECT_EQ(gtaSA.results[0].code, "ASNAEB");
    EXPECT_EQ(gtaSA.results[0].jamcrc, 0x555fc201);
}

TEST(GTA_SA_STDTHREAD, basic_calc_base_3) {
    GTA_SA_STDTHREAD gtaSA;
    gtaSA.minRange = 181961000;
    gtaSA.maxRange = 181961100;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 1);
    EXPECT_EQ(gtaSA.results[0].index, 181961057);
    EXPECT_EQ(gtaSA.results[0].code, "OHDUDE");
    EXPECT_EQ(gtaSA.results[0].jamcrc, 0xe958788a);
}

TEST(GTA_SA_STDTHREAD, basic_calc_base_4) {
    GTA_SA_STDTHREAD gtaSA;
    gtaSA.minRange = 299376700;
    gtaSA.maxRange = 299376800;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 1);
    EXPECT_EQ(gtaSA.results[0].index, 299376767);
    EXPECT_EQ(gtaSA.results[0].code, "YECGAA");
    EXPECT_EQ(gtaSA.results[0].jamcrc, 0x40cf761);
}

TEST(GTA_SA_STDTHREAD, basic_calc_base_5) {
    GTA_SA_STDTHREAD gtaSA;
    gtaSA.minRange = 20810700;
    gtaSA.maxRange = 147491500;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 3);
    EXPECT_EQ(gtaSA.results[0].index, 20810792);
    EXPECT_EQ(gtaSA.results[0].code, "ASNAEB");
    EXPECT_EQ(gtaSA.results[0].jamcrc, 0x555fc201);

    EXPECT_EQ(gtaSA.results[1].index, 75396850);
    EXPECT_EQ(gtaSA.results[1].code, "FHYSTV");
    EXPECT_EQ(gtaSA.results[1].jamcrc, 0x44b34866);

    EXPECT_EQ(gtaSA.results[2].index, 147491485);
    EXPECT_EQ(gtaSA.results[2].code, "LJSPQK");
    EXPECT_EQ(gtaSA.results[2].jamcrc, 0xfeda77f7);
}

#if defined(_OPENMP)
TEST(GTA_SA_OPENMP, basic_calc_mode_openmp) {
    GTA_SA_OPENMP gtaSA;
    gtaSA.minRange = 0;
    gtaSA.maxRange = 60000;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 0);
}
#endif

#if defined(BUILD_WITH_CUDA)

TEST(GTA_SA_CUDA, basic_calc_cuda_1) {
    GTA_SA_CUDA gtaSA;
    gtaSA.minRange = 0;
    gtaSA.maxRange = 6000000;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 0);
}

TEST(GTA_SA_CUDA, basic_calc_cuda_2) {
    GTA_SA_CUDA gtaSA;
    gtaSA.minRange = 20810700;
    gtaSA.maxRange = 20810800;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 1);
    EXPECT_EQ(gtaSA.results[0].index, 20810792);
    EXPECT_EQ(gtaSA.results[0].code, "ASNAEB");
    EXPECT_EQ(gtaSA.results[0].jamcrc, 0x555fc201);
}

TEST(GTA_SA_CUDA, basic_calc_cuda_3) {
    GTA_SA_CUDA gtaSA;
    gtaSA.minRange = 181961000;
    gtaSA.maxRange = 181961100;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 1);
    EXPECT_EQ(gtaSA.results[0].index, 181961057);
    EXPECT_EQ(gtaSA.results[0].code, "OHDUDE");
    EXPECT_EQ(gtaSA.results[0].jamcrc, 0xe958788a);
}

TEST(GTA_SA_CUDA, basic_calc_cuda_4) {
    GTA_SA_CUDA gtaSA;
    gtaSA.minRange = 299376700;
    gtaSA.maxRange = 299376800;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 1);
    EXPECT_EQ(gtaSA.results[0].index, 299376767);
    EXPECT_EQ(gtaSA.results[0].code, "YECGAA");
    EXPECT_EQ(gtaSA.results[0].jamcrc, 0x40cf761);
}

/*
TEST(GTA_SA_CUDA, basic_calc_cuda_5) {
    GTA_SA_CUDA gtaSA;
    gtaSA.minRange = 0;
    gtaSA.maxRange = 8'031'810'176;

    gtaSA.run();

    EXPECT_EQ(gtaSA.results.size(), 152);
    EXPECT_EQ(gtaSA.results[0].index, 20810792);
    EXPECT_EQ(gtaSA.results[0].code, "ASNAEB");
    EXPECT_EQ(gtaSA.results[0].jamcrc, 0x555fc201);

    EXPECT_EQ(gtaSA.results[50].index, 2746380464);
    EXPECT_EQ(gtaSA.results[50].code, "HWCWJYV");
    EXPECT_EQ(gtaSA.results[50].jamcrc, 0xb2Afe368);

    EXPECT_EQ(gtaSA.results[100].index, 5151099438);
    EXPECT_EQ(gtaSA.results[100].code, "PQNCSOD");
    EXPECT_EQ(gtaSA.results[100].jamcrc, 0x680416b1);

    EXPECT_EQ(gtaSA.results[125].index, 6668091073);
    EXPECT_EQ(gtaSA.results[125].code, "UOETDAG");
    EXPECT_EQ(gtaSA.results[125].jamcrc, 0xd4966d59);

    EXPECT_EQ(gtaSA.results[151].index, 8003059504);
    EXPECT_EQ(gtaSA.results[151].code, "YWOBEJX");
    EXPECT_EQ(gtaSA.results[151].jamcrc, 0xe1ef01ea);
}
*/

#endif

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
