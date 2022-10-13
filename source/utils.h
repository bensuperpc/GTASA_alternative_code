#ifndef UTILS_H
#define UTILS_H

#ifndef __FUNCTION_NAME__
#  ifdef WIN32  // WINDOWS
#    define __FUNCTION_NAME__ __FUNCTION__
#  else  //*NIX
#    define __FUNCTION_NAME__ __func__
#  endif
#endif

#define MIN(a, b) (((a) < (b)) ? a : b)
#define MAX(a, b) (((a) > (b)) ? a : b)

#endif  // UTILS_H
