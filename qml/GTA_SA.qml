import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material

import org.bensuperpc.GTA_SAObjectType 1.0
import org.bensuperpc.GTA_SAObjects 1.0
//import org.bensuperpc.TableModelObjects 1.0

import "custom/"

Page {
    title: qsTr("BruteForce GTA SA")
    id: page

    Flickable {
        anchors.fill: parent
        anchors.centerIn: parent
        width: parent.width
        height: parent.height
        anchors.margins: 5
        id: flickable

        contentHeight: Math.max(grid.implicitHeight, parent.height)
        contentWidth: Math.max(grid.Layout.minimumWidth, parent.width)
        flickableDirection: Flickable.AutoFlickIfNeeded

        GridLayout {
            id: grid
            anchors.fill: parent
            anchors.margins: 5
            columns: 12
            rows: 12

            ProportionalRect {
                Layout.columnSpan: 12
                Layout.rowSpan: 3
                implicitWidth: 100
                GridLayout {
                    anchors.fill: parent
                    anchors.margins: 5
                    columns: 2
                    rows: 1
                    ProportionalRect {
                        Layout.columnSpan: 1
                        Layout.rowSpan: 1
                        GroupBox {
                            title: qsTr("BruteForce GTA SA")
                            anchors.fill: parent
                            implicitHeight: rightBoxLayout2.implicitHeight
                            implicitWidth: rightBoxLayout2.implicitWidth
                            ColumnLayout {
                                id: rightBoxLayout2
                                RowLayout {
                                    TextField {
                                        Layout.fillWidth: true
                                        id: minRangeValue
                                        placeholderText: qsTr("Enter minimal range value")
                                        text: GTA_SASingleton.minRangeValue.toLocaleString(
                                                'fullwide', {
                                                    "useGrouping": false
                                                })
                                        selectByMouse: true
                                        validator: RegularExpressionValidator {
                                            regularExpression: /[0-9]+/
                                        }
                                    }
                                    Binding {
                                        target: GTA_SASingleton
                                        property: "minRangeValue"
                                        value: minRangeValue.text
                                    }
                                }
                                RowLayout {
                                    TextField {
                                        id: maxRangeValue
                                        Layout.fillWidth: true
                                        placeholderText: qsTr("Enter maximum range value")
                                        text: GTA_SASingleton.maxRangeValue.toLocaleString(
                                                'fullwide', {
                                                    "useGrouping": false
                                                })
                                        selectByMouse: true
                                        validator: RegularExpressionValidator {
                                            regularExpression: /[0-9]+/
                                        }
                                    }
                                    Binding {
                                        target: GTA_SASingleton
                                        property: "maxRangeValue"
                                        value: maxRangeValue.text
                                    }
                                }
                                RowLayout {
                                    Button {
                                        id: launchRunner
                                        //Layout.alignment: Qt.AlignHCenter
                                        Layout.fillWidth: true
                                        // text: qsTr("Launch Bruteforce")
                                        text: GTA_SASingleton.buttonValue
                                        onClicked: GTA_SASingleton.runOp()
                                    }
                                }
                            }
                        }
                    }
                    ProportionalRect {
                        Layout.columnSpan: 1
                        Layout.rowSpan: 1
                        GroupBox {
                            title: qsTr("Settings")
                            anchors.fill: parent
                            ColumnLayout {
                                RowLayout {
                                    RadioButton {
                                        id: enableSTDTHREAD
                                        enabled: true
                                        checked: (GTA_SASingleton.calc_mode == 0 ? true : false)
                                        text: qsTr("std::thread")
                                        onToggled: {
                                            GTA_SASingleton.set_calc_mode(0)
                                        }
                                    }
                                    RadioButton {
                                        id: enableOpenMP
                                        checked: (GTA_SASingleton.calc_mode == 1 ? true : false)
                                        enabled: GTA_SASingleton.builtWithOpenMP
                                        text: qsTr("OpenMP")
                                        onToggled: {
                                            GTA_SASingleton.set_calc_mode(1)
                                        }
                                    }
                                    RadioButton {
                                        id: enableCUDA
                                        enabled: GTA_SASingleton.builtWithCUDA
                                        checked: (GTA_SASingleton.calc_mode == 2 ? true : false)
                                        text: qsTr("CUDA")
                                        onToggled: {
                                            GTA_SASingleton.set_calc_mode(2)
                                        }
                                    }
                                }
                                RowLayout {
                                    // enabled: (GTA_SASingleton.builtWithOpenMP ? enableOpenMP.checkState: false)
                                    Label {
                                        text: qsTr("CPU core: ")
                                    }
                                    Slider {
                                        id: nbr_thread_value
                                        value: GTA_SASingleton.nbrThreadValue
                                        stepSize: 1
                                        from: 1
                                        to: GTA_SASingleton.maxThreadSupport()
                                        snapMode: Slider.SnapAlways
                                    }

                                    Binding {
                                        target: GTA_SASingleton
                                        property: "nbrThreadValue"
                                        value: nbr_thread_value.value
                                    }
                                    Label {
                                        text: (GTA_SASingleton.nbrThreadValue
                                            >= 10) ? GTA_SASingleton.nbrThreadValue : " "
                                                        + GTA_SASingleton.nbrThreadValue
                                    }
                                }
                                RowLayout {
                                    Label {
                                        text: qsTr("Block size: ")
                                    }
                                    Slider {
                                        id: cuda_block_size_slider
                                        enabled: GTA_SASingleton.builtWithCUDA
                                        // value: 1024
                                        stepSize: 64
                                        from: 64
                                        to: 1024
                                        snapMode: Slider.SnapAlways
                                    }
                                    Binding {
                                        target: GTA_SASingleton
                                        property: "cudaBlockSize"
                                        value: cuda_block_size_slider.value
                                    }
                                    Label {
                                        text: (GTA_SASingleton.cudaBlockSize
                                            >= 100) ? GTA_SASingleton.cudaBlockSize : " "
                                                        + GTA_SASingleton.cudaBlockSize
                                    }
                                }
                            }
                        }
                    }
                }
            }

            ProportionalRect {
                Layout.columnSpan: 12
                Layout.rowSpan: 6
                GridLayout {
                    anchors.fill: parent
                    anchors.margins: 5
                    columns: 1
                    rows: 1
                    ProportionalRect {
                        Layout.columnSpan: 1
                        Layout.rowSpan: 1
                        GroupBox {
                            title: qsTr("Result")
                            anchors.fill: parent
                            /*
                            TableView {
                                anchors.fill: parent
                                columnSpacing: 1
                                rowSpacing: 1
                                clip: true
                                ScrollIndicator.horizontal: ScrollIndicator {}
                                ScrollIndicator.vertical: ScrollIndicator {}
                                model: TableModelObjects
                                delegate: Rectangle {
                                    implicitWidth: 164
                                    implicitHeight: 20
                                    border.color: "black"
                                    border.width: 2
                                    color: heading ? 'antiquewhite' : "aliceblue"
                                    Text {
                                        text: tabledata
                                        font.pointSize: 10
                                        font.bold: heading ? true : false
                                        anchors.centerIn: parent
                                    }
                                }
                            }
                            */
                        }
                    }
                }
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
                            title: qsTr("Export to JSON")
                            anchors.fill: parent

                            Button {
                                Layout.alignment: Qt.AlignHCenter
                                text: qsTr("Export data")
                                enabled: false
                            }
                        }
                    }
                    ProportionalRect {
                        Layout.columnSpan: 1
                        Layout.rowSpan: 1
                        GroupBox {
                            title: qsTr("Export to CSV")
                            anchors.fill: parent

                            Button {
                                text: qsTr("Export data")
                                enabled: false
                            }
                        }
                    }
                    ProportionalRect {
                        Layout.columnSpan: 1
                        Layout.rowSpan: 1
                        GroupBox {
                            title: qsTr("Export to TXT")
                            anchors.fill: parent

                            Button {
                                text: qsTr("Export data")
                                enabled: false
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
