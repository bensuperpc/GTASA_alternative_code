#ifndef GTA_SA_OPENMP_HPP
#define GTA_SA_OPENMP_HPP

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

#include "BS_thread_pool.hpp"
#include "GTA_SA_cheat_finder_virtual.hpp"

#if __has_include("omp.h")
#include <omp.h>
#endif

class GTA_SA_OPENMP final : public GTA_SA_Virtual {
   public:
    explicit GTA_SA_OPENMP();
    ~GTA_SA_OPENMP();

    GTA_SA_OPENMP& operator=(const GTA_SA_OPENMP& other);

    void inline runner(const std::uint64_t i) override;

    void run() override;
};

#endif  // GTA_SA_OPENMP_HPP
