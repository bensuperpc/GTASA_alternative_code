#ifndef GTASARESULT_HPP
#define GTASARESULT_HPP

#include <cstdint>  // std::uint64_t, std::uint32_t
#include <string>   // std::string

class GTASAResult {
   public:
    explicit GTASAResult(std::uint64_t, std::string&, std::uint32_t, std::uint64_t);
    explicit GTASAResult(std::uint64_t, std::string&&, std::uint32_t, std::uint64_t);

    //TODO: Removed it after rewrite
    explicit GTASAResult(std::uint64_t, std::string, std::uint32_t, std::string);
    
    GTASAResult();
    ~GTASAResult();

    GTASAResult& operator=(const GTASAResult&) noexcept;
    bool operator==(const GTASAResult& other) const noexcept;

    std::uint64_t index = 0;
    std::string code = "";
    std::uint32_t jamcrc = 0;
    std::uint64_t codeIndex = 0;

    std::string associated_code = "";
};

#endif  // GTASARESULT_HPP
