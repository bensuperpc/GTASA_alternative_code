#ifndef UTILS_CPP_HPP
#define UTILS_CPP_HPP

#define KB 1024u
#define MB 1024u * 1024u
#define GB 1024u * 1024u * 1024u
#define TB 1024u * 1024u * 1024u * 1024u

#define QMLREGISTERSINGLETONTYPE(variable, uri, versionMajor, versionMinor, typeName)       \
    qmlRegisterSingletonType<decltype(variable)>(uri, versionMajor, versionMinor, typeName, \
                                                 [&](QQmlEngine*, QJSEngine*) -> QObject* { return &variable; });

// In C++20: https://en.cppreference.com/w/cpp/utility/source_location
#ifndef __FUNCTION_NAME__
#ifdef WIN32  // WINDOWS
#define __FUNCTION_NAME__ __FUNCTION__
#else  // UNIX
#define __FUNCTION_NAME__ __func__
#endif
#endif

/*
#if __cplusplus >= 202002L
#include <source_location>
#include <string>
#include <iostream>
void debugLog(const std::string& message, const std::source_location& location = std::source_location::current())
{
    std::cout << "Debug: "
              << location.file_name() << ':'
              << location.line() << ' '
              << message << std::endl;
}
#endif
*/

#endif  // UTILS_H
