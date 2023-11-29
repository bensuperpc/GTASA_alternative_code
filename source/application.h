#ifndef APPLICATION_H
#define APPLICATION_H

#include <QDebug>
#include <QJSEngine>
#include <QObject>
#include <QQmlEngine>
#include <QString>
#include <QStringView>
#include <QtConcurrent>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_AMD64)
#if __has_include("x86intrin.h")
#include <x86intrin.h>
#endif
#endif

#include "utils/about_compilation.h"

class application final : public QObject {
    Q_OBJECT
    Q_DISABLE_COPY(application)

    Q_PROPERTY(QString textBoxValue READ textBoxValue WRITE setTextBoxValue NOTIFY textBoxValueChanged)

   public:
    explicit application(QObject* parent = nullptr);

    QString textBoxValue() const;

    // Compiler Info
    Q_INVOKABLE QString compilerArch();
    Q_INVOKABLE QString compilerName();
    Q_INVOKABLE QString compilerVersion();
    Q_INVOKABLE QString cxxVersion();
    Q_INVOKABLE QString cVersion();
    Q_INVOKABLE QString osType();
    Q_INVOKABLE QString osTypeFull();
    Q_INVOKABLE QString buildDate();
    Q_INVOKABLE QString buildType();
    Q_INVOKABLE QString qtVersion();
    Q_INVOKABLE QString appAuthor();
    Q_INVOKABLE QString appBusiness();
    Q_INVOKABLE QString appVersion();

   private:
    QString _textBoxValue = "Hello, world!";
   public slots:
    void setTextBoxValue(QString value);
   signals:
    void textBoxValueChanged(QString value);
};

#endif  // APPLICATION_H
