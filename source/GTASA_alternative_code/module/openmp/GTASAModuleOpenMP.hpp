#ifndef GTASAMODULEOPENMP_HPP
#define GTASAMODULEOPENMP_HPP

#include <module/GTASAModule.hpp>

#if __has_include("omp.h")
#include <omp.h>
#endif

class GTASAModuleOpenMP final : public GTASAModule {
   public:
    explicit GTASAModuleOpenMP();
    ~GTASAModuleOpenMP();

    auto run(std::uint64_t startRange, std::uint64_t endRange) -> std::vector<GTASAResult> override final;

    private:
     auto runner(const std::uint64_t i) const -> GTASAResult override final;
};

#endif  // GTASAMODULEOPENMP_HPP
