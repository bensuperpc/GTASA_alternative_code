import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material

import org.bensuperpc.GTA_SAObjectType 1.0
import org.bensuperpc.GTA_SAObjects 1.0
import org.bensuperpc.TableModelObjects 1.0

Page {
    title: qsTr("BruteForce GTA SA")
    id: page

    Flickable {
        anchors.fill: parent
        anchors.margins: 10
        id: flickable

        contentHeight: columnLayout.implicitHeight
        contentWidth: columnLayout.implicitWidth
        flickableDirection: Flickable.AutoFlickIfNeeded

        ColumnLayout {
            id: columnLayout
            spacing: 10
            width: page.width
            // height: children.height
            RowLayout {
                Layout.alignment: Qt.AlignHCenter
                GroupBox {
                    title: qsTr("BruteForce GTA SA")
                    ColumnLayout {
                        RowLayout {
                            TextField {
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
                        Layout.alignment: Qt.AlignHCenter
                        // text: qsTr("Launch Bruteforce")
                        text: GTA_SASingleton.buttonValue
                        onClicked: GTA_SASingleton.runOp()
                    }
                }
            }
        }
        GroupBox {
            title: qsTr("Settings")
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
                        to: GTA_SASingleton.max_thread_support()
                        snapMode: Slider.SnapAlways
                    }

                    Binding {
                        target: GTA_SASingleton
                        property: "nbrThreadValue"
                        value: nbr_thread_value.value
                    }
                    Label {
                        text: (GTA_SASingleton.nbrThreadValue
                        >= 10) ? GTA_SASingleton.nbrThreadValue : " " + GTA_SASingleton.nbrThreadValue
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
                        property: "cuda_block_size"
                        value: cuda_block_size_slider.value
                    }
                    Label {
                        text: (GTA_SASingleton.cuda_block_size
                        >= 100) ? GTA_SASingleton.cuda_block_size : " "
                        + GTA_SASingleton.cuda_block_size
                    }
                }
            }
        }
    }
    RowLayout {
        Layout.alignment: Qt.AlignHCenter
        GroupBox {
            title: qsTr("Result")
            Layout.alignment: Qt.AlignHCenter
            ColumnLayout {
                RowLayout {
                    Layout.alignment: Qt.AlignHCenter
                    TableView {
                        width: 480
                        height: 380
                        columnSpacing: 1
                        rowSpacing: 1
                        clip: true
                        ScrollIndicator.horizontal: ScrollIndicator {}
                        ScrollIndicator.vertical: ScrollIndicator {}
                        model: TableModelObjects
                        delegate: Rectangle {
                            implicitWidth: 120
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
                }
            }
        }
    }

    RowLayout {
        Layout.alignment: Qt.AlignHCenter
        GroupBox {
            title: qsTr("Export to JSON")
            Layout.alignment: Qt.AlignHCenter
            ColumnLayout {
                RowLayout {
                    Layout.alignment: Qt.AlignHCenter
                    Button {
                        Layout.alignment: Qt.AlignHCenter
                        text: qsTr("Export data")
                        enabled: false
                    }
                }
            }
        }
        GroupBox {
            title: qsTr("Export to CSV")
            Layout.alignment: Qt.AlignHCenter
            ColumnLayout {
                RowLayout {
                    Layout.alignment: Qt.AlignHCenter
                    Button {
                        Layout.alignment: Qt.AlignHCenter
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
