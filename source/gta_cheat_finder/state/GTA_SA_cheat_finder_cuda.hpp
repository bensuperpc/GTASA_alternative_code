#ifndef GTA_SA_CUDA_HPP
#define GTA_SA_CUDA_HPP

#include <cstring>      // strlen
#include <iomanip>      // std::setw
#include <iostream>     // std::cout
#include <string>       // std::string
#include <string_view>  // std::string_view

#include "GTA_SA_cheat_finder_virtual.hpp"

class GTA_SA_CUDA final : public GTA_SA_Virtual {
   public:
    explicit GTA_SA_CUDA();
    ~GTA_SA_CUDA();

    GTA_SA_CUDA& operator=(const GTA_SA_CUDA& other);

    void inline runner(const std::uint64_t i) override;

    void run() override;
};

#endif  // GTA_SA_CUDA_HPP
