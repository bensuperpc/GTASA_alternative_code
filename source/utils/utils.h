#ifndef UTILS_H
#define UTILS_H

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

#endif  // UTILS_H
