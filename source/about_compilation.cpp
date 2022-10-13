#include "about_compilation.h"

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
