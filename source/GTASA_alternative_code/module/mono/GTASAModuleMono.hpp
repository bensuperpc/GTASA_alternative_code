#ifndef GTASAMODULEMONO_HPP
#define GTASAMODULEMONO_HPP

#include <module/GTASAModule.hpp>

class GTASAModuleMono final : public GTASAModule {
   public:
    explicit GTASAModuleMono();
    ~GTASAModuleMono();

    auto run(std::uint64_t startRange, std::uint64_t endRange) -> std::vector<GTASAResult> override final;
    
    private:
     auto runner(const std::uint64_t i) const -> GTASAResult override final;
};

#endif  // GTA_SA_STDTHREAD_HPP
