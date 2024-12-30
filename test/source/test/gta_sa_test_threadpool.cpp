#include <string>

#include "gtest/gtest.h"

#include "GTASAEngine.hpp"

TEST(GTASAEngine, basic_calc_threadpool_1) {
    GTASAEngine gtaSA = GTASAEngine();
    GTASARequest* request = gtaSA.addRequest("THREADPOOL", 0, 6'000'000);
    gtaSA.waitAllRequests();

    EXPECT_EQ(request->getResults().size(), 0);
}


TEST(GTASAEngine, basic_calc_threadpool_2) {
    GTASAEngine gtaSA = GTASAEngine();
    GTASARequest* request = gtaSA.addRequest("THREADPOOL", 20'810'700, 20'810'800);
    gtaSA.waitAllRequests();

    EXPECT_EQ(request->getResults().size(), 1);
    EXPECT_EQ(request->getResults()[0].index, 20810792);
    EXPECT_EQ(request->getResults()[0].code, "ASNAEB");
    EXPECT_EQ(request->getResults()[0].jamcrc, 0x555fc201);
}


TEST(GTASAEngine, basic_calc_threadpool_3) {
    GTASAEngine gtaSA = GTASAEngine();
    GTASARequest* request = gtaSA.addRequest("THREADPOOL", 181'961'000, 181'961'100);
    gtaSA.waitAllRequests();

    EXPECT_EQ(request->getResults().size(), 1);
    EXPECT_EQ(request->getResults()[0].index, 181'961'057);
    EXPECT_EQ(request->getResults()[0].code, "OHDUDE");
    EXPECT_EQ(request->getResults()[0].jamcrc, 0xe958788a);
}

TEST(GTASAEngine, basic_calc_threadpool_4) {
    GTASAEngine gtaSA = GTASAEngine();
    GTASARequest* request = gtaSA.addRequest("THREADPOOL", 299'376'700, 299'376'800);
    gtaSA.waitAllRequests();

    EXPECT_EQ(request->getResults().size(), 1);
    EXPECT_EQ(request->getResults()[0].index, 299'376'767);
    EXPECT_EQ(request->getResults()[0].code, "YECGAA");
    EXPECT_EQ(request->getResults()[0].jamcrc, 0x40cf761);
}

/*
TEST(GTASAEngine, basic_calc_threadpool_5) {
    GTASAEngine gtaSA = GTASAEngine();
    GTASARequest* request = gtaSA.addRequest("THREADPOOL", 0, 5'200'000'000);
    gtaSA.waitAllRequests();

    EXPECT_EQ(request->getResults().size(), 102);
    EXPECT_EQ(request->getResults()[0].index, 20'810'792);
    EXPECT_EQ(request->getResults()[0].code, "ASNAEB");
    EXPECT_EQ(request->getResults()[0].jamcrc, 0x555fc201);

    EXPECT_EQ(request->getResults()[50].index, 2'746'380'464);
    EXPECT_EQ(request->getResults()[50].code, "HWCWJYV");
    EXPECT_EQ(request->getResults()[50].jamcrc, 0xb2Afe368);

    EXPECT_EQ(request->getResults()[100].index, 5'151'099'438);
    EXPECT_EQ(request->getResults()[100].code, "PQNCSOD");
    EXPECT_EQ(request->getResults()[100].jamcrc, 0x680416b1);
}
*/

TEST(GTASAEngine, multi_calc_threadpool_1) {
    GTASAEngine gtaSA = GTASAEngine();
    GTASARequest* request1 = gtaSA.addRequest("THREADPOOL", 0, 182'000'000);
    GTASARequest* request2 = gtaSA.addRequest("THREADPOOL", 182'000'000, 300'000'000);
    GTASARequest* request3 = gtaSA.addRequest("THREADPOOL", 300'000'000, 545'000'000);
    gtaSA.waitAllRequests();

    EXPECT_EQ(request1->getResults().size(), 5);
    EXPECT_EQ(request1->getResults()[0].index, 20'810'792);
    EXPECT_EQ(request1->getResults()[0].code, "ASNAEB");
    EXPECT_EQ(request1->getResults()[0].jamcrc, 0x555fc201);

    EXPECT_EQ(request1->getResults()[4].index, 181'961'057);
    EXPECT_EQ(request1->getResults()[4].code, "OHDUDE");
    EXPECT_EQ(request1->getResults()[4].jamcrc, 0xe958788a);

    EXPECT_EQ(request2->getResults().size(), 4);
    EXPECT_EQ(request2->getResults()[0].index, 198'489'210);
    EXPECT_EQ(request2->getResults()[0].code, "PRIEBJ");
    EXPECT_EQ(request2->getResults()[0].jamcrc, 0xf2aa0c1d);

    EXPECT_EQ(request2->getResults()[3].index, 299'376'767);
    EXPECT_EQ(request2->getResults()[3].code, "YECGAA");
    EXPECT_EQ(request2->getResults()[3].jamcrc, 0x40cf761);

    EXPECT_EQ(request3->getResults().size(), 4);
    EXPECT_EQ(request3->getResults()[0].index, 311'365'503);
    EXPECT_EQ(request3->getResults()[0].code, "ZEIIVG");
    EXPECT_EQ(request3->getResults()[0].jamcrc, 0x74d4fcb1);

    EXPECT_EQ(request3->getResults()[3].index, 535'721'682);
    EXPECT_EQ(request3->getResults()[3].code, "ASBHGRB");
    EXPECT_EQ(request3->getResults()[3].jamcrc, 0xa7613f99);
}

auto main(int argc, char** argv) -> int {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
