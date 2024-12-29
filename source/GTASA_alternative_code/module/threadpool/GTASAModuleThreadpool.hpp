#ifndef GTASAMODULETHREADPOOL_HPP
#define GTASAMODULETHREADPOOL_HPP

#include <module/GTASAModule.hpp>

#include "BS_thread_pool.hpp"

class GTASAModuleThreadpool final : public GTASAModule {
   public:
    explicit GTASAModuleThreadpool();
    ~GTASAModuleThreadpool();

    auto run(std::uint64_t startRange, std::uint64_t endRange) -> std::vector<GTASAResult> override final;
    
    private:
     auto runner(const std::uint64_t i) const -> GTASAResult override final;
     BS::thread_pool<BS::tp::none> _pool = BS::thread_pool<BS::tp::none>(0);
};

#endif  // GTA_SA_STDTHREAD_HPP
