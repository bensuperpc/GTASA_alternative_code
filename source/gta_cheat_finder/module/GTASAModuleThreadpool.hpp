#ifndef GTASAMODULETHREADPOOL_HPP
#define GTASAMODULETHREADPOOL_HPP

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
