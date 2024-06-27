#ifndef GTA_SA_STDTHREAD_HPP
#define GTA_SA_STDTHREAD_HPP

#include <algorithm>    // std::find
#include <array>        // std::array
#include <chrono>       // std::chrono
#include <cmath>        // std::ceil
#include <cstring>      // strlen
#include <iomanip>      // std::setw
#include <iostream>     // std::cout
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::pair
#include <utility>      // std::make_pair
#include <vector>       // std::vector

// #define BS_THREAD_POOL_ENABLE_PAUSE
// #define BS_THREAD_POOL_ENABLE_PRIORITY
// #define BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK
#include "BS_thread_pool.hpp"

#include "GTA_SA_cheat_finder_virtual.hpp"

class GTA_SA_STDTHREAD final : public GTA_SA_Virtual {
   public:
    explicit GTA_SA_STDTHREAD();
    ~GTA_SA_STDTHREAD();

    GTA_SA_STDTHREAD& operator=(const GTA_SA_STDTHREAD& other);

    void inline runner(const std::uint64_t i) override;

    void run() override;
};

#endif  // GTA_SA_STDTHREAD_HPP
