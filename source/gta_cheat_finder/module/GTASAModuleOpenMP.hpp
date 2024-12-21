#ifndef GTASAMODULEOPENMP_HPP
#define GTASAMODULEOPENMP_HPP

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

#if __has_include("omp.h")
#include <omp.h>
#endif

class GTASAModuleOpenMP final : public GTASAModuleVirtual {
   public:
    explicit GTASAModuleOpenMP();
    ~GTASAModuleOpenMP();

    std::vector<GTASAResult> run(std::uint64_t startRange, std::uint64_t endRange) override;

    private:
     GTASAResult runner(const std::uint64_t i);
};

#endif  // GTASAMODULEOPENMP_HPP
