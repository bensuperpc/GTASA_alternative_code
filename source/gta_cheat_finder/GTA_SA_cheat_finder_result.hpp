#ifndef GTA_SA_RESULT_HPP
#define GTA_SA_RESULT_HPP

#include <cstdint>  // std::uint64_t, std::uint32_t
#include <string>   // std::string

class result {
   public:
    explicit result(std::uint64_t, std::string, std::uint32_t, std::string);
    ~result();

    result& operator=(const result&) noexcept;
    bool operator==(const result& other) const noexcept;

    std::uint64_t index = 0;
    std::string code = "";
    std::uint32_t jamcrc = 0;
    std::string associated_code;
};

#endif  // GTA_SA_RESULT_HPP
