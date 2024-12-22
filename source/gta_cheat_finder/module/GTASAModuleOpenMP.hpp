#ifndef GTASAMODULEOPENMP_HPP
#define GTASAMODULEOPENMP_HPP

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
