/**
 * @file about_compilation.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2022-04-25
 *
 * MIT License
 *
 */

// More info: https://sourceforge.net/p/predef/wiki/Compilers/

#include "compilation.hpp"

namespace my
{
namespace compile
{

const std::string arch()
{
#if defined(__i386__) || defined(_M_IX86)
  return std::string("i386");
#elif defined(__x86_64__) || defined(_M_X64)
  return std::string("x86_64");
#elif defined(__arm__) || defined(_M_ARM)
  return std::string("arm");
#elif defined(__aarch64__) || defined(_M_ARM64)
  return std::string("arm64");
#elif defined(__PPC64__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
  return std::string("ppc64le");
#elif defined(__PPC64__)
  return std::string("ppc64");
#elif defined(__MIPS__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
  return std::string("mipsel");
#elif defined(__MIPS__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
  return std::string("mips");
#elif defined(__PPC__)
  return std::string("ppc");
#elif defined(__AVR__)
  return std::string("avr");
#endif
}

const std::string compiler()
{
#if defined(__GNUC__) && !defined(__MINGW64__) && !defined(__MINGW32__)
  return std::string("GCC");
#elif defined(__clang__)
  return std::string("Clang");
#elif defined(_MSC_VER)
  return std::string("MSVC");
#elif defined(__INTEL_COMPILER)
  return std::string("icc");
#elif defined(__OPEN64__)
  return std::string("Open64");
#elif defined(__BORLANDC__)
  return std::string("Borland");
#elif defined(__DECC_VER) || defined(__DECCXX_VER)
  return std::string("Compaq");
#elif defined(__HP_aCC) || defined(__HP_cc)
  return std::string("HP");
#elif defined(__DCC__)
  return std::string("Diab");
#elif defined(__IBMCPP__)
  return std::string("IBM");
#elif defined(__xlC__)
  return std::string("xlC");
#elif defined(__PGI)
  return std::string("PGI");
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  return std::string("Sun");
#elif defined(__FCC_VERSION)
  return std::string("Fujitsu");
#elif defined(__ghs__)
  return std::string("Green Hill");
#elif defined(__KCC)
  return std::string("KAI");
#elif defined(__COMO__)
  return std::string("Comeau");
#elif defined(__MINGW32__)
  return std::string("MinGW");
#elif defined(__CYGWIN__)
  return std::string("Cygwin");
#elif defined(__MINGW64__)
  return std::string("MinGW64");
#elif defined(_MRI)
  return std::string("Microtec");
#elif defined(__ICL)
  return std::string("Intel");
#elif defined(__MRC__)
  return std::string("MPW");
#elif defined(__SC__)
  return std::string("Symantec");
#elif defined(__MWERKS__)
  return std::string("Metrowerks");
#elif defined(__PGI)
  return std::string("Portland Group");
#elif defined(_PACC_VER)
  return std::string("Palm");
#elif defined(__RENESAS__)
  return std::string("Renesas");
#elif defined(__TURBOC__)
  return std::string("Turbo C");
#elif defined(__TI_COMPILER_VERSION__)
  return std::string("Texas Instruments");
#elif defined(__WATCOMC__)
  return std::string("Watcom");
#else
  return std::string("unknown");
#endif
}

const std::string compiler_ver()
{
#if defined(__GNUC__)
  return std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(__clang__)
  return std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "."
      + std::to_string(__clang_patchlevel__);
#elif defined(_MSC_VER) || defined(_MSC_FULL_VER)
  return std::to_string(_MSC_FULL_VER);
#elif defined(__INTEL_COMPILER)
  return std::to_string(__INTEL_COMPILER);
#elif defined(__BORLANDC__)
  return std::to_string(__BORLANDC__);
#elif defined(__DECCXX_VER)
  return std::to_string(__DECCXX_VER);
#elif defined(__ghs__)
  return std::to_string(__GHS_VERSION_NUMBER__);
#elif defined(__xlC__) || defined(__IBMCPP__)
  return std::to_string(__IBMCPP__);
#elif defined(__MRC__)
  return std::to_string(__MRC__);
#elif defined(__OPEN64__)
  return std::to_string(__OPEN64__);
#elif defined(_PACC_VER)
  return std::to_string(_PACC_VER);
#elif defined(__WATCOMC__)
  return std::to_string(__WATCOMC__);
#else
  return std::string("unknown");
#endif
}

const std::string os()
{
// https://stackoverflow.com/a/5920028/10152334
#if defined(_WIN32)
  return std::string("win");
#elif defined(__APPLE__)
  return std::string("mac");
#elif defined(__unix__)
  return std::string("unix");
#else
  return std::string("unknown");
#endif
}

const std::string cxx()
{
  return std::to_string(__cplusplus);
}

const std::string build_date()
{
  return std::string(__DATE__) + " : " + std::string(__TIME__);
}

const std::string arduino()
{
#if defined(TEENSYDUINO)
#  if defined(__AVR_ATmega32U4__)
  return std::string("Teensy 2.0");
#  elif defined(__AVR_AT90USB1286__)
  return std::string("Teensy++ 2.0");
#  elif defined(__MK20DX128__)
  return std::string("Teensy 3.0");
#  elif defined(__MK20DX256__)
  return std::string("Teensy 3.2");  // and Teensy 3.1 (obsolete)
#  elif defined(__MKL26Z64__)
  return std::string("Teensy LC");
#  elif defined(__MK64FX512__)
  return std::string("Teensy 3.5");
#  elif defined(__MK66FX1M0__)
  return std::string("Teensy 3.6");
#  else
  return std::string("Unknown board");
#  endif
#else
#  if defined(ARDUINO_AVR_ADK)
  return std::string("Mega Adk");
#  elif defined(ARDUINO_AVR_BT)  // Bluetooth
  return std::string("Bt");
#  elif defined(ARDUINO_AVR_DUEMILANOVE)
  return std::string("Duemilanove");
#  elif defined(ARDUINO_AVR_ESPLORA)
  return std::string("Esplora");
#  elif defined(ARDUINO_AVR_ETHERNET)
  return std::string("Ethernet");
#  elif defined(ARDUINO_AVR_FIO)
  return std::string("Fio");
#  elif defined(ARDUINO_AVR_GEMMA)
  return std::string("Gemma");
#  elif defined(ARDUINO_AVR_LEONARDO) || defined(__AVR_ATmega32U4__) || defined(__AVR_ATmega16U4__)
  return std::string("Leonardo");
#  elif defined(ARDUINO_AVR_LILYPAD)
  return std::string("Lilypad");
#  elif defined(ARDUINO_AVR_LILYPAD_USB)
  return std::string("Lilypad Usb");
#  elif defined(ARDUINO_AVR_MEGA) || defined(__AVR_ATmega1280__)
  return std::string("Mega");
#  elif defined(ARDUINO_AVR_MEGA2560) || defined(__AVR_ATmega2560__)
  return std::string("Mega 2560");
#  elif defined(ARDUINO_AVR_MICRO)
  return std::string("Micro");
#  elif defined(ARDUINO_AVR_MINI)
  return std::string("Mini");
#  elif defined(ARDUINO_AVR_NANO)
  return std::string("Nano");
#  elif defined(ARDUINO_AVR_NG)
  return std::string("NG");
#  elif defined(ARDUINO_AVR_PRO)
  return std::string("Pro");
#  elif defined(ARDUINO_AVR_ROBOT_CONTROL)
  return std::string("Robot Ctrl");
#  elif defined(ARDUINO_AVR_ROBOT_MOTOR)
  return std::string("Robot Motor");
#  elif defined(ARDUINO_AVR_UNO) || defined(__AVR_ATmega328P__) || defined(__AVR_ATmega168__)
  return std::string("Uno");
#  elif defined(ARDUINO_AVR_YUN)
  return std::string("Yun");
#  elif defined(ARDUINO_SAM_DUE)
  return std::string("Due");
#  elif defined(ARDUINO_SAMD_ZERO)
  return std::string("Zero");
#  elif defined(ARDUINO_ARC32_TOOLS)
#    define BOARD "101");
#  else
  return std::string("Unknown board");
#  endif
#endif
}

}  // namespace compile
}  // namespace my
