#ifndef GTASAMODULECUDA_HPP
#define GTASAMODULECUDA_HPP

#include <module/GTASAModule.hpp>

class GTASAModuleCUDA final : public GTASAModule {
   public:
    explicit GTASAModuleCUDA();
    ~GTASAModuleCUDA();

    auto run(std::uint64_t startRange, std::uint64_t endRange) -> std::vector<GTASAResult> override final;

    private:
     uint64_t _cudaBlockSize = 64;
};

#endif  // GTASAMODULECUDA_HPP
