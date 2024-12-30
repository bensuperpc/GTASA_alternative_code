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
        contentHeight: height
        contentWidth: width
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
                Layout.rowSpan: 12
                //Layout.row: 0
                //Layout.column: 0
                GroupBox {
                    title: qsTr("QML/C++ binding")
                    anchors.fill: parent
                    padding: 6
                }
            }

            ProportionalRect {
                Layout.columnSpan: 12
                Layout.rowSpan: 4

                GroupBox {
                    title: qsTr("Listview from C++")
                    anchors.fill: parent
                    padding: 2

                    ColumnLayout {
                        width: parent.width
                        clip: true
                        RowLayout {
                            Layout.alignment: Qt.AlignHCenter
                            ListView {
                                Layout.minimumWidth: 200
                                Layout.preferredHeight: 100
                                Layout.alignment: Qt.AlignHCenter
                                //boundsBehavior: Flickable.DragOverBounds
                                snapMode: ListView.SnapToItem
                                model: monModele
                                delegate: Component {
                                    Switch {
                                        text: modelData
                                        height: 35
                                        hoverEnabled: false
                                    }
                                }
                            }
                            Item {
                                Layout.minimumWidth: 200
                                Layout.preferredHeight: 75
                                Component {
                                    id: monDelegue
                                    Rectangle {
                                        width: 250
                                        height: 25
                                        anchors.horizontalCenter: parent.horizontalCenter
                                        color: "darkgrey"
                                        Text {
                                            text: "From C++ : " + modelData
                                        }
                                    }
                                }
                                ListView {
                                    id: maVue
                                    model: monModele
                                    delegate: monDelegue
                                    anchors.fill: parent
                                }
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
