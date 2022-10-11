import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material
import MyApp.Images

Page {
    title: qsTr("About Page")
    id: page

    GridLayout {
        id: main_grid
        anchors.fill: parent
        columns: 10
        rows: 10

        component ProportionalRect: Rectangle {
            Layout.fillHeight: true
            Layout.fillWidth: true
            Layout.preferredWidth: Layout.columnSpan
            Layout.preferredHeight: Layout.rowSpan
            color: "transparent"
        }

        ProportionalRect {
            Layout.columnSpan: 5
            Layout.rowSpan: 5

            GroupBox {
                title: qsTr("App and Compiler")
                anchors.fill: parent
                Column {
                    Text {
                        text: qsTr("Compiler: " + about_compilation.return_Compiler_name(
                                       ))
                        color: "white"
                        font.bold: true
                        fontSizeMode: Text.Fit
                        minimumPixelSize: 5
                        font.pixelSize: 12
                    }
                    Text {
                        text: qsTr("Compiler vers: " + about_compilation.return_Compiler_version(
                                       ))
                        color: "white"
                        font.bold: true
                        fontSizeMode: Text.Fit
                        minimumPixelSize: 5
                        font.pixelSize: 12
                    }
                    Text {
                        text: qsTr("C++ version: " + about_compilation.return_Cplusplus_used(
                                       ))
                        color: "white"
                        font.bold: true
                        fontSizeMode: Text.Fit
                        minimumPixelSize: 5
                        font.pixelSize: 12
                    }
                    Text {
                        text: qsTr("Build: " + about_compilation.return_BuildDate(
                                       ))
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
            Layout.columnSpan: 5
            Layout.rowSpan: 5
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
                        text: qsTr(
                                  "OpenMP: " + about_compilation.openmpIsEnable(
                                      ))
                        color: "white"
                        font.bold: true
                        fontSizeMode: Text.Fit
                        minimumPixelSize: 5
                        font.pixelSize: 12
                    }
                    Text {
                        text: qsTr("Nvidia CUDA: " + about_compilation.cudaIsEnable(
                                       ))
                        color: "white"
                        font.bold: true
                        fontSizeMode: Text.Fit
                        minimumPixelSize: 5
                        font.pixelSize: 12
                    }
                    Text {
                        text: qsTr(
                                  "OpenCL: " + about_compilation.openclIsEnable(
                                      ))
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
            Layout.columnSpan: 5
            Layout.rowSpan: 5
            GroupBox {
                title: qsTr("About this app")
                // Layout.alignment: Qt.AlignHCenter
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
                        text: "Source: <a href='https://github.com/Bensuperpc/KrackX'>Click here</a>"
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
        ProportionalRect {
            Layout.columnSpan: 5
            Layout.rowSpan: 5
            GroupBox {
                title: qsTr("Test")
                // Layout.alignment: Qt.AlignHCenter
                anchors.fill: parent
                    Image {
                        source: Qt.resolvedUrl("/bensuperpc.com/qml_files/img/Profile_400x400.jpg")
                        anchors.fill: parent
                        sourceSize.width: parent.width
                        sourceSize.height: parent.height
                    }
                    /*
                    Rectangle {
                        width: parent.width
                        height: parent.height
                        color: "transparent"
                        LiveImage {
                            anchors.fill: parent
                            // width: 480
                            // height: 480
                            enable_rescale: true
                            image: LiveImageProvider.image
                        }
                    }
                    */
            }
        }
    }
}
