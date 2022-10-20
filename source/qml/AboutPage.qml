import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material

import org.bensuperpc.ABCObjects 1.0

Page {
    title: qsTr("About Page")
    id: page

    GridLayout {
        anchors.fill: parent
        anchors.margins: 5
        columns: 12
        rows: 12

        component ProportionalRect: Rectangle {
            Layout.fillHeight: true
            Layout.fillWidth: true
            Layout.preferredWidth: Layout.columnSpan
            Layout.preferredHeight: Layout.rowSpan
            color: "transparent"
        }

        ProportionalRect {
            Layout.columnSpan: 12
            Layout.rowSpan: 3

            GridLayout {
                anchors.fill: parent
                anchors.margins: 5
                columns: 3
                rows: 1

                ProportionalRect {
                    Layout.columnSpan: 1
                    Layout.rowSpan: 1

                    GroupBox {
                        title: qsTr("App and Compiler")
                        anchors.fill: parent
                        Column {
                            Text {
                                text: qsTr("Compiler: " + ABCObjects.return_Compiler_name(
                                                ))
                                color: "white"
                                font.bold: true
                                fontSizeMode: Text.Fit
                                minimumPixelSize: 5
                                font.pixelSize: 12
                            }

                            Text {
                                text: qsTr("Compiler vers: " + ABCObjects.return_Compiler_version(
                                                ))
                                color: "white"
                                font.bold: true
                                fontSizeMode: Text.Fit
                                minimumPixelSize: 5
                                font.pixelSize: 12
                            }

                            Text {
                                text: qsTr("C++ version: " + ABCObjects.return_Cplusplus_used(
                                                ))
                                color: "white"
                                font.bold: true
                                fontSizeMode: Text.Fit
                                minimumPixelSize: 5
                                font.pixelSize: 12
                            }
                            Text {
                                text: qsTr("Build: " + ABCObjects.return_BuildDate())
                                color: "white"
                                font.bold: true
                                fontSizeMode: Text.Fit
                                minimumPixelSize: 5
                                font.pixelSize: 12
                            }
                        }
                    }
                }

                ProportionalRect {
                    Layout.columnSpan: 1
                    Layout.rowSpan: 1

                    GroupBox {
                        title: qsTr("Libs")
                        anchors.fill: parent
                        Column {
                            Text {
                                text: qsTr("Qt version: " + qtversion)
                                color: "white"
                                font.bold: true
                                fontSizeMode: Text.Fit
                                minimumPixelSize: 5
                                font.pixelSize: 12
                            }

                            Text {
                                text: qsTr("OpenMP: " + ABCObjects.openmpIsEnable())
                                color: "white"
                                font.bold: true
                                fontSizeMode: Text.Fit
                                minimumPixelSize: 5
                                font.pixelSize: 12
                            }

                            Text {
                                text: qsTr("Nvidia CUDA: " + ABCObjects.cudaIsEnable())
                                color: "white"
                                font.bold: true
                                fontSizeMode: Text.Fit
                                minimumPixelSize: 5
                                font.pixelSize: 12
                            }

                            Text {
                                text: qsTr("OpenCL: " + ABCObjects.openclIsEnable())
                                color: "white"
                                font.bold: true
                                fontSizeMode: Text.Fit
                                minimumPixelSize: 5
                                font.pixelSize: 12
                            }
                        }
                    }
                }

                ProportionalRect {
                    Layout.columnSpan: 1
                    Layout.rowSpan: 1

                    GroupBox {
                        title: qsTr("About this app")
                        anchors.fill: parent

                        Column {
                            Text {
                                text: qsTr("Created by: <a href='https://github.com/Bensuperpc'>Bensuperpc</a>")
                                color: "white"
                                font.bold: true
                                fontSizeMode: Text.Fit
                                minimumPixelSize: 5
                                font.pixelSize: 12
                                wrapMode: Text.WordWrap
                                onLinkActivated: Qt.openUrlExternally(link)

                                MouseArea {
                                    anchors.fill: parent
                                    acceptedButtons: Qt.NoButton
                                    cursorShape: parent.hoveredLink ? Qt.PointingHandCursor : Qt.ArrowCursor
                                }
                            }
                            Text {
                                text: "Source: <a href='https://github.com/bensuperpc/GTA_SA_cheat_finder'>Click here</a>"
                                color: "white"
                                font.bold: true
                                fontSizeMode: Text.Fit
                                minimumPixelSize: 5
                                font.pixelSize: 12
                                wrapMode: Text.WordWrap
                                onLinkActivated: Qt.openUrlExternally(link)

                                MouseArea {
                                    anchors.fill: parent
                                    acceptedButtons: Qt.NoButton
                                    cursorShape: parent.hoveredLink ? Qt.PointingHandCursor : Qt.ArrowCursor
                                }
                            }
                        }
                    }
                }
            }
        }

        ProportionalRect {
            Layout.columnSpan: 12
            Layout.rowSpan: 3

            // 2 columns inside second row
            GridLayout {
                anchors.fill: parent
                anchors.margins: 5
                columns: 2
                rows: 1

                ProportionalRect {
                    Layout.columnSpan: 1
                    Layout.rowSpan: 1
                    Image {
                        source: Qt.resolvedUrl(
                                    "/bensuperpc.com/qml_files/img/Profile_400x400.jpg")
                        anchors.fill: parent
                        sourceSize.width: parent.width
                        sourceSize.height: parent.height
                    }
                }

                ProportionalRect {
                    Layout.columnSpan: 1
                    Layout.rowSpan: 1
                    Text {
                        color: "white"
                        font.bold: true
                        fontSizeMode: Text.Fit
                        minimumPixelSize: 5
                        font.pixelSize: 12
                        wrapMode: Text.WordWrap
                        text: "I'm Bensuperpc"
                    }
                }
            }
        }

        ProportionalRect {
            Layout.columnSpan: 12
            Layout.rowSpan: 3
            color: "blue"

            // 1 column inside third row
            GridLayout {
                anchors.fill: parent
                anchors.margins: 5
                columns: 1
                rows: 1

                ProportionalRect {
                    Layout.columnSpan: 1
                    Layout.rowSpan: 1
                    color: "red"
                }
            }
        }

        ProportionalRect {
            Layout.columnSpan: 12
            Layout.rowSpan: 3
            color: "yellow"

            // 3 columns inside fourth row
            GridLayout {
                anchors.fill: parent
                anchors.margins: 5
                columns: 3
                rows: 1

                ProportionalRect {
                    Layout.columnSpan: 1
                    Layout.rowSpan: 1
                    color: "red"
                }

                ProportionalRect {
                    Layout.columnSpan: 1
                    Layout.rowSpan: 1
                    color: "green"
                }

                ProportionalRect {
                    Layout.columnSpan: 1
                    Layout.rowSpan: 1
                    color: "blue"
                }
            }
        }

        /*
        ProportionalRect {
            Layout.columnSpan: 5
            Layout.rowSpan: 5
            GroupBox {
                title: qsTr("Test")
                // Layout.alignment: Qt.AlignHCenter
                anchors.fill: parent
                Image {
                    source: Qt.resolvedUrl(
                                "/bensuperpc.com/qml_files/img/Profile_400x400.jpg")
                    anchors.fill: parent
                    sourceSize.width: parent.width
                    sourceSize.height: parent.height
                }
            }
        }
        */
    }
}
