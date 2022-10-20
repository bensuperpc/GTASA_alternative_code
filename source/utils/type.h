#ifndef BENLIB_UTILS_TYPE_HPP_
#define BENLIB_UTILS_TYPE_HPP_

#include <cstddef>  // std::size_t

#ifdef defined(__cplusplus) || defined(c_plusplus)

constexpr uint64_t KB = 1024u;
constexpr uint64_t MB = 1024u * 1024u;
constexpr uint64_t GB = 1024u * 1024u * 1024u;

constexpr std::size_t operator""_KB(const unsigned long long v)
{
  return 1024u * v;
}

constexpr std::size_t operator""_MB(const unsigned long long v)
{
  return 1024u * 1024u * v;
}

constexpr std::size_t operator""_GB(const unsigned long long v)
{
  return 1024u * 1024u * 1024u * v;
}

#else

#define KB 1024u
#define MB 1024u * 1024u
#define GB 1024u * 1024u * 1024u

#define _KB(v) 1024u * v
#define _MB(v) 1024u * 1024u * v
#define _GB(v) 1024u * 1024u * 1024u * v

#endif

#endif  // UTILS_H
