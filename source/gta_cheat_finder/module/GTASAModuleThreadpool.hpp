#ifndef GTASAMODULETHREADPOOL_HPP
#define GTASAMODULETHREADPOOL_HPP

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

#include "GTASAModuleVirtual.hpp"

#include "BS_thread_pool.hpp"

class GTASAModuleThreadpool final : public GTASAModuleVirtual {
   public:
    explicit GTASAModuleThreadpool();
    ~GTASAModuleThreadpool();

    std::vector<GTASAResult> run(std::uint64_t startRange, std::uint64_t endRange) override;
    
    private:
     GTASAResult runner(const std::uint64_t i);
     BS::thread_pool<BS::tp::none> _pool = BS::thread_pool<BS::tp::none>(0);
};

#endif  // GTA_SA_STDTHREAD_HPP
