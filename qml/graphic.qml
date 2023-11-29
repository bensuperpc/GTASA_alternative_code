import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Window
import QtCharts

import org.bensuperpc.lineseries 1.0
import org.bensuperpc.application 1.0

import "custom/"

Page {
    title: qsTr("About")
    id: page

    Flickable {
        id: flickable
        anchors.fill: parent

        anchors.leftMargin: 5
        anchors.rightMargin: 5
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

                //Layout.row: 0
                //Layout.column: 0
                //color: "red"
                GroupBox {
                    title: qsTr("Graphic (From C++)")
                    anchors.fill: parent
                    padding: 2

                    ChartView {
                        id: chartView
                        animationOptions: ChartView.NoAnimation
                        anchors.fill: parent
                        theme: window.Material.theme === Material.Dark ? ChartView.ChartThemeDark : ChartView.ChartThemeLight
                        ValueAxis {
                            id: axisY1
                            min: -1
                            max: 4
                        }

                        ValueAxis {
                            id: axisY2
                            min: -10
                            max: 5
                        }

                        ValueAxis {
                            id: axisX
                            min: 0
                            max: 1024
                        }

                        LineSeries {
                            id: lineSeries1
                            name: "signal 1"
                            axisX: axisX
                            axisY: axisY1
                        }
                        LineSeries {
                            id: lineSeries2
                            name: "signal 2"
                            axisX: axisX
                            axisYRight: axisY2
                        }
                        Timer {
                            id: refreshTimer
                            interval: 1 / 30 * 1000 // 30 Hz
                            running: true
                            repeat: true
                            onTriggered: {
                                UIData.update(chartView.series(0))
                                UIData.update(chartView.series(1))
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
