#ifndef GTASAMODULECUDA_HPP
#define GTASAMODULECUDA_HPP

#include <module/GTASAModule.hpp>

class GTASAModuleCUDA final : public GTASAModule {
   public:
    explicit GTASAModuleCUDA();
    ~GTASAModuleCUDA();

    std::vector<GTASAResult> run(std::uint64_t startRange, std::uint64_t endRange) override;

    private:
     uint64_t _cudaBlockSize = 64;
};

#endif  // GTASAMODULECUDA_HPP
