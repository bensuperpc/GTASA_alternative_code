#include "about_compilation.h"

#if __has_include(<QApplication>)

about_compilation::about_compilation(QObject* parent)
    : QObject(parent)
{
}

string about_compilation::ver_string(int a, int b, int c)
{
  std::ostringstream ss;
  ss << a << '.' << b << '.' << c;
  return ss.str();
}

#else
// Not building with Qt
#endif
