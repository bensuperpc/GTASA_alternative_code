#ifndef GTA_SA_OPENCL_HPP
#define GTA_SA_OPENCL_HPP

#include "GTA_SA_cheat_finder_virtual.hpp"

class GTA_SA_OPENCL final : public GTA_SA_Virtual {
   public:
    explicit GTA_SA_OPENCL();
    ~GTA_SA_OPENCL();

    GTA_SA_OPENCL& operator=(const GTA_SA_OPENCL& other);

    void inline runner(const std::uint64_t i) override;
    COMPUTE_TYPE type() const override;

    void run() override;
};

#endif  // GTA_SA_OPENCL_HPP
