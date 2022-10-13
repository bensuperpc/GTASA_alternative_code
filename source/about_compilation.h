#ifndef ABOUT_COMPILATION_H
#define ABOUT_COMPILATION_H

#include <QDateTime>
#include <QObject>
#include <QString>
#include <iostream>
#include <sstream>
#include <string>

#include "compilation.hpp"

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#  if __has_include("cuda.h")
#    include <cuda.h>
#    ifndef BUILD_WITH_CUDA
#      define BUILD_WITH_CUDA
#      define BUILD_WITH_CUDA
#    endif
#  else
#    if _MSC_VER && !__INTEL_COMPILER
#      pragma message("Can t find cuda.h, disable CUDA module")
#    else
#      warning Can t find cuda.h, disable CUDA module.
#    endif
#  endif
#endif

using namespace std;

class about_compilation : public QObject
{
  Q_OBJECT
public:
  explicit about_compilation(QObject* parent = nullptr);

  std::string ver_string(int a, int b, int c);

  Q_INVOKABLE
  QString return_Compiler_version()
  {
    return QString::fromStdString(my::compile::compiler_ver());
  }

  Q_INVOKABLE
  QString return_Compiler_name()
  {
    return QString::fromStdString(my::compile::compiler());
  }

  Q_INVOKABLE
  QString return_Cplusplus_used()
  {
    return QString::fromStdString(my::compile::cxx());
  }

  Q_INVOKABLE
  QString return_BuildDate()
  {
    return QString::fromStdString(my::compile::build_date());
  }
  Q_INVOKABLE
  QString openmpIsEnable()
  {
#if !defined(_OPENMP)
    return QString::fromStdString("false");
#else
    return QString::fromStdString("true");
#endif
  }
  Q_INVOKABLE
  QString cudaIsEnable()
  {
#if !defined(BUILD_WITH_CUDA)
    return QString::fromStdString("false");
#else
    return QString::fromStdString("true");
#endif
  }
  Q_INVOKABLE
  QString openclIsEnable()
  {
    return QString::fromStdString("false");
  }

signals:

public slots:
};

#endif  // ABOUT_COMPILATION_H
