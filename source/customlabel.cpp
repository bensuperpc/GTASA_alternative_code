#include "customlabel.h"

MyLabel::MyLabel(QObject* parent) : QObject{parent} {}

void MyLabel::setMyObject(QObject* obj) {
    if (!obj) {
        return;
    }

    myObject = obj;
    myObject->setProperty("visible", QVariant(true));
    myObject->setProperty("text", QVariant("Hello world from C++ !"));

    QFont&& font = myObject->property("font").value<QFont>();
    font.setPointSize(18);
    myObject->setProperty("font", font);

    QVariant returnedValue;
    QVariant message = "Hello world!";

    QMetaObject::invokeMethod(myObject, "myQMLFunction", Q_RETURN_ARG(QVariant, returnedValue), Q_ARG(QVariant, message));

    qDebug() << "QML function returned:" << returnedValue.toString();
}
