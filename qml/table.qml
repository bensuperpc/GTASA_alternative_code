import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Window

import org.bensuperpc.TableData 1.0

import "custom/"

Page {
    title: qsTr("About")
    id: page

    Flickable {
        id: flickable
        anchors.fill: parent
        width: parent.width
        height: parent.height

        anchors.leftMargin: 5
        anchors.rightMargin: 4
        anchors.topMargin: 5
        anchors.bottomMargin: 5

        //contentHeight: gridLayout.height
        //contentWidth: gridLayout.width
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

                GroupBox {
                    title: qsTr("Table (From C++)")
                    anchors.fill: parent
                    padding: 2

                    TableView {
                        anchors.fill: parent
                        columnSpacing: 1
                        rowSpacing: 1
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

        ScrollIndicator.vertical: ScrollIndicator {}
        ScrollIndicator.horizontal: ScrollIndicator {}
        ScrollBar.vertical: ScrollBar {}
        ScrollBar.horizontal: ScrollBar {}
    }
}
