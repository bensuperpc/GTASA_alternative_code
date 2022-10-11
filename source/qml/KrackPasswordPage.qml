import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material
import QtCharts

Page {
    title: qsTr("Main Page")
    id: page
    property int timeStep: 0

    Flickable {
        anchors.fill: parent
        contentHeight: columnLayout.implicitHeight
        contentWidth: columnLayout.implicitWidth
        flickableDirection: Flickable.AutoFlickIfNeeded

        ColumnLayout {
            // unique child
            id: columnLayout
            spacing: 10
            width: page.width // ensure correct width
            height: children.height // ensure correct height

            ComboBox {
                id: comboBox1
                Layout.alignment: Qt.AlignHCenter
                model: myApp.comboList
                editable: false
                onAccepted: {
                    if (editableCombo.find(currentText) === -1) {
                        model.append({
                                         "text": editText
                                     })
                        currentIndex = editableCombo.find(editText)
                    }
                }
            }

            Button {
                id: button1
                Layout.alignment: Qt.AlignHCenter
                text: qsTr("Remove Item")
                onClicked: myApp.removeElement(comboBox1.currentIndex)
            }

            TextField {
                id: textEdit1
                Layout.alignment: Qt.AlignHCenter
                text: qsTr("Text Edit")
                font.pixelSize: 12
            }

            Button {
                id: button2
                Layout.alignment: Qt.AlignHCenter
                text: qsTr("Add Item")
                onClicked: myApp.addElement(textEdit1.text)
            }
            TextField {
                id: textfield1
                Layout.alignment: Qt.AlignHCenter
                placeholderText: qsTr("Enter name 1")
                text: myApp.author
                onTextChanged: {
                    myApp.author = textfield1.text
                    //textfield2.text = textfield1.text
                }
            }
            TextField {
                id: textfield2
                Layout.alignment: Qt.AlignHCenter
                placeholderText: qsTr("Enter name 2")
                text: myApp.author
                onTextChanged: {
                    myApp.author = textfield2.text
                    //textfield2.text = textfield1.text
                }
            }

            ChartView {
                id: chartView
                Layout.alignment: Qt.AlignHCenter
                antialiasing: true
                width: page.width * 0.8
                height: page.height * 0.6

                ValueAxis {
                    id: axisX
                    min: 0
                    max: 200
                }

                Component.onCompleted: {
                    mapper.series = series2
                }

                LineSeries {
                    id: series1
                    axisX: axisX
                    name: "From QML"
                }

                LineSeries {
                    id: series2
                    axisX: axisX
                    name: "From C++"
                }
            }

            Timer {
                interval: 100
                repeat: true
                running: true
                onTriggered: {
                    timeStep++
                    var y = (1 + Math.cos(timeStep / 10.0)) / 2.0
                    series1.append(timeStep, y)
                }
            }
        }

        ScrollIndicator.vertical: ScrollIndicator {}
        ScrollIndicator.horizontal: ScrollIndicator {}
        ScrollBar.vertical: ScrollBar {}
        ScrollBar.horizontal: ScrollBar {}
    }
}
