import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Window

import org.bensuperpc.GTA_SAObjectType 1.0
import org.bensuperpc.GTA_SAObjects 1.0
//import org.bensuperpc.TableModelObjects 1.0

import org.bensuperpc.TableData 1.0

Page {
    title: qsTr("About")
    id: page
    component Box : Rectangle {
        property string text
        property int fontSize: 30
        property int alignment: Text.AlignHCenter
        color: "#33b5e5"
        implicitHeight: txt.implicitHeight + 10
        implicitWidth: txt.implicitWidth + 10
        Layout.fillWidth: true
        Layout.minimumWidth: implicitWidth
        Text {
            id: txt
            text: parent.text
            color: "white"
            anchors.fill: parent
            anchors.margins: 5
            font.pixelSize: parent.fontSize
            horizontalAlignment: parent.alignment
        }
    }

    Flickable {
        anchors.fill: parent
        contentHeight: Math.max(grid.implicitHeight, parent.height)
        contentWidth: Math.max(grid.Layout.minimumWidth, parent.width)

        GridLayout {
            id: grid
            anchors.left: parent.left
            anchors.right: parent.right
            columns: (window.width > 900) ? 3 : (window.width > 660) ? 2 : 1;

            ColumnLayout {
                id: leftBox
                Layout.fillHeight: true
                Layout.margins: 10
                Layout.alignment: Qt.AlignTop
                Box {
                    implicitHeight: rightBoxLayout.implicitHeight
                    implicitWidth: rightBoxLayout.implicitWidth
                    Layout.fillHeight: true
                    color: "blue"
                    GroupBox {
                        title: qsTr("BruteForce GTA SA")
                        id: rightBoxLayout
                        anchors.fill: parent
                        ColumnLayout {
                            anchors.fill: parent
                            RowLayout {
                                TextField {
                                    id: minRangeValue
                                    Layout.fillWidth: true
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
            }

            ColumnLayout {
                id: leftBox2
                Layout.fillHeight: true
                Layout.margins: 10
                Layout.alignment: Qt.AlignTop
                Box {
                    implicitHeight: rightBoxLayout2.implicitHeight
                    implicitWidth: rightBoxLayout2.implicitWidth
                    color: "green"
                    Layout.fillHeight: true
                    GroupBox {
                        title: qsTr("Settings")
                        id: rightBoxLayout2
                        anchors.fill: parent
                        ColumnLayout {
                            anchors.fill: parent
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
            }

            ColumnLayout {
                id: leftBox4
                Layout.fillHeight: true
                Layout.margins: 10
                Layout.alignment: Qt.AlignTop
                Box {
                    implicitHeight: rightBoxLayout3.implicitHeight
                    implicitWidth: rightBoxLayout3.implicitWidth
                    color: "darkorange"
                    GroupBox {
                        title: qsTr("Settings")
                        id: rightBoxLayout3
                        anchors.fill: parent
                        ColumnLayout {
                            anchors.fill: parent
                            RowLayout {
                                ComboBox {
                                    model: ["JSON", "CSV", "TXT"]
                                }
                                Button {
                                    text: qsTr("Export data")
                                    enabled: false
                                }
                            }
                        }
                    }
                }
            }

            Box {
                id: result
                color: "red"
                Layout.columnSpan: grid.columns
                Layout.margins: 10
                implicitHeight: tableResult.implicitHeight
                implicitWidth: tableResult.implicitWidth
                GroupBox {
                    title: qsTr("Result")
                    id: tableResult
                    anchors.fill: parent
                    ColumnLayout {
                        anchors.fill: parent
                        RowLayout {
                            TableView {
                                clip: true
                                id: tableViewData
                                
                                selectionModel: ItemSelectionModel {}

                                ScrollIndicator.horizontal: ScrollIndicator {}
                                ScrollIndicator.vertical: ScrollIndicator {}

                                model: TableDataModel
                                delegate: Component {
                                    Rectangle {
                                        implicitWidth: {
                                            var caseWidth = (tableViewData.width - 2)
                                                    / TableDataModel.columnCount()
                                            if (caseWidth > 80 && caseWidth < 200) {
                                                return caseWidth
                                            } else {
                                                return 80
                                            }
                                        }

                                        implicitHeight: 20
                                        border.color: window.Material.theme
                                                    === Material.Dark ? "black" : "black"
                                        border.width: 2
                                        color: heading ? "antiquewhite" : "aliceblue"
                                        Text {
                                            text: tabledata
                                            font.pointSize: 10
                                            font.bold: heading ? true : false
                                            anchors.centerIn: parent
                                        }
                                        
                                        MouseArea {
                                            anchors.fill: parent
                                            onClicked: {
                                                console.log("Clicked on " + tabledata)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            /*
            Box {
                id: result2
                color: "transparent"
                Layout.columnSpan: grid.columns
                GridLayout {
                    anchors.fill: parent
                    anchors.margins: 5
                    GroupBox {
                        title: qsTr("Export to JSON")
                        Layout.fillWidth: true
                        Button {
                            text: qsTr("Export data")
                            enabled: false
                        }
                    }
                    
                    GroupBox {
                        title: qsTr("Export to CSV")
                        Layout.fillWidth: true
                        Button {
                            text: qsTr("Export data")
                            enabled: false
                        }
                    }

                    GroupBox {
                        title: qsTr("Export to TXT")
                        Layout.fillWidth: true
                        Button {
                            text: qsTr("Export data")
                            enabled: false
                        }
                    }
                }
            }
            */
        }
    }

}
