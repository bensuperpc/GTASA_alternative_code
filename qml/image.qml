import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Window
import QtCharts

import org.bensuperpc.application 1.0

import AppLib 1.0

Page {
    title: qsTr("Home")
    id: page

    Flickable {
        id: flickable
        anchors.fill: parent

        anchors.leftMargin: 5
        anchors.rightMargin: 5
        anchors.topMargin: 5
        anchors.bottomMargin: 5

        // contentHeight: gridLayout.height
        // contentWidth: gridLayout.width
        // contentWidth: Math.max(gridLayout.width, 540)
        // contentHeight: Math.max(gridLayout.height, 960)
        flickableDirection: Flickable.AutoFlickIfNeeded

        GridLayout {
            id: gridLayout
            anchors.fill: parent
            anchors.margins: 1

            columnSpacing: 1
            rowSpacing: 1
            columns: 12
            rows: 12

            ProportionalRect {
                Layout.columnSpan: 12
                Layout.rowSpan: 10
                //Layout.row: 0
                //Layout.column: 0
                GroupBox {
                    title: qsTr("Image (From QML and C++)")
                    anchors.fill: parent
                    padding: 2

                    Flickable {
                        anchors.fill: parent
                        contentWidth: width
                        contentHeight: flow.implicitHeight
                        clip: true
                        flickableDirection: Flickable.AutoFlickIfNeeded
                        Flow {
                            width: parent.width
                            anchors.margins: 1
                            spacing: 1
                            id: flow
                            Repeater {
                                model: enableCppImages.checked ? cppImageCount.value : 0
                                Image {
                                    source: "image://async/cat_sticking_out_its_tongue.jpg"
                                    // Layout.fillHeight: true
                                    // Layout.preferredWidth: parent.width * 0.5
                                    fillMode: Image.PreserveAspectFit
                                    width: 64
                                    height: 64
                                    asynchronous: true
                                    smooth: false
                                }
                            }
                            Repeater {
                                model: enableQMLImages.checked ? qmlImageCount.value : 0

                                Image {
                                    source: Qt.resolvedUrl(
                                                "/bensuperpc.org/bensuperpc/image/cat_sticking_out_its_tongue.jpg")
                                    width: 64
                                    height: 64
                                    fillMode: Image.PreserveAspectFit
                                    asynchronous: true
                                    smooth: false
                                }
                            }
                        }
                        ScrollIndicator.vertical: ScrollIndicator {}
                        ScrollBar.vertical: ScrollBar {}
                    }
                }
            }
            ProportionalRect {
                Layout.columnSpan: 12
                Layout.rowSpan: 2
                GroupBox {
                    title: qsTr("Settings")
                    anchors.fill: parent
                    padding: 2

                    ColumnLayout {
                        anchors.fill: parent
                        RowLayout {
                            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
                            Switch {
                                id: enableCppImages
                                text: qsTr("C++ Image")
                                checked: true
                            }
                            Slider {
                                id: cppImageCount
                                value: 16
                                stepSize: 16
                                from: 0
                                to: 128
                                snapMode: Slider.SnapAlways
                            }
                            Label {
                                text: qsTr("C++ image: %1").arg(
                                          cppImageCount.value)
                            }
                        }
                        RowLayout {
                            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
                            Switch {
                                id: enableQMLImages
                                text: qsTr("QML Image")
                                checked: true
                            }
                            Slider {
                                id: qmlImageCount
                                value: 16
                                stepSize: 16
                                from: 0
                                to: 128
                                snapMode: Slider.SnapAlways
                            }
                            Label {
                                text: qsTr("C++ image: %1").arg(
                                          qmlImageCount.value)
                            }
                        }
                    }
                }
            }
        }

        ScrollIndicator.vertical: ScrollIndicator {}
        ScrollIndicator.horizontal: ScrollIndicator {}
        ScrollBar.vertical: ScrollBar {}
        ScrollBar.horizontal: ScrollBar {}
    }
}
