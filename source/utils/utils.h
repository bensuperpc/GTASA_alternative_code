#ifndef UTILS_H
#define UTILS_H

// In C++20: https://en.cppreference.com/w/cpp/utility/source_location
#ifndef __FUNCTION_NAME__
#  ifdef WIN32  // WINDOWS
#    define __FUNCTION_NAME__ __FUNCTION__
#  else  // UNIX
#    define __FUNCTION_NAME__ __func__
#  endif
#endif

#define MIN(a, b) (((a) < (b)) ? a : b)
#define MAX(a, b) (((a) > (b)) ? a : b)

#define ABS(a) (((a) < 0) ? -(a) : (a))

// From https://github.com/RobLoach/raylib-cpp/blob/master/include/raylib-cpp-utils.hpp
#ifndef GETTERSETTER
/**
 * A utility to build get and set methods on top of a property.
 *
 * @param type The type of the property.
 * @param method The human-readable name for the method.
 * @param name The machine-readable name of the property.
 */
#  define GETTERSETTER(type, method, name) \
    /** Retrieves the name value for the object. @return The name value of the object. */ \
    inline type get_##method() const \
    { \
      return name; \
    } \
    /** Sets the name value for the object. @param value The value of which to set name to. */ \
    inline void set_##method(type value) \
    { \
      name = value; \
    }
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
}
#endif

#endif  // UTILS_H
