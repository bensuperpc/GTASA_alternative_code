#include "application.h"

application::application(QObject* parent) : QObject{parent} {}

QString application::textBoxValue() const {
    return _textBoxValue;
}

void application::setTextBoxValue(QString value) {
    qDebug() << "setTextBoxValue"
             << ": " << value;
    if (value == _textBoxValue)
        return;
    _textBoxValue = value;
    emit textBoxValueChanged(value);
}

QString application::compilerArch() {
    return QString::fromStdString(compile::arch());
}

QString application::compilerName() {
    return QString::fromStdString(compile::compiler());
}

QString application::compilerVersion() {
    return QString::fromStdString(compile::compiler_ver());
}

QString application::cxxVersion() {
    return QString::fromStdString(compile::cxx());
}

QString application::cVersion() {
    return QString::fromStdString(compile::c());
}

QString application::osType() {
    return QString::fromStdString(compile::os());
}

QString application::osTypeFull() {
    return QString::fromStdString(compile::os_adv());
}

QString application::buildDate() {
    return QString::fromStdString(compile::build_date());
}

QString application::buildType() {
    return QString::fromStdString(compile::build_type());
}

QString application::qtVersion() {
    return QString::fromStdString(compile::qt_version());
}

QString application::appVersion() {
#ifdef PROJECT_VERSION
    return QString(PROJECT_VERSION);
#else
    return QString("Unknown");
#endif
}

QString application::appAuthor() {
    return QString("Bensuperpc");
}

QString application::appBusiness() {
    return QString("bensuperpc (bensuperpc.org)");
}
