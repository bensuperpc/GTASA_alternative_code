import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Window

import org.bensuperpc.application 1.0

import "custom/"

Page {
    title: qsTr("Swipe")
    id: page

    SwipeView {
        id: view

        currentIndex: 0
        anchors.fill: parent
        Repeater {
            model: 3
            Pane {
                id: pagePane
                GroupBox {
                    title: qsTr("Page N°%1").arg(view.currentIndex + 1)
                    anchors.fill: parent
                    anchors.bottomMargin: 6
                    padding: 2

                    Flickable {
                        anchors.fill: parent
                        contentWidth: width
                        contentHeight: flickablePage.implicitHeight
                        clip: true
                        Flow {
                            id: flickablePage
                            width: parent.width
                            spacing: 0

                            ColumnLayout {
                                width: parent.width
                                spacing: 0
                                ItemDelegate {
                                    Layout.fillWidth: true
                                    spacing: 0
                                    padding: 0

                                    contentItem: RowLayout {
                                        spacing: 0
                                        Layout.fillWidth: true
                                        Label {
                                            text: qsTr("Test %1a").arg(
                                                      view.currentIndex + 1)
                                            Layout.fillWidth: true
                                            leftPadding: 32
                                        }

                                        Label {
                                            text: "OK"
                                            rightPadding: 16
                                            color: "Green"
                                        }
                                    }
                                }
                                ItemDelegate {
                                    Layout.fillWidth: true
                                    spacing: 0
                                    padding: 0

                                    contentItem: RowLayout {
                                        spacing: 0
                                        Layout.fillWidth: true
                                        Label {
                                            text: qsTr("Test %1b").arg(
                                                      view.currentIndex + 1)
                                            Layout.fillWidth: true
                                            leftPadding: 32
                                        }

                                        Label {
                                            text: "WARNING"
                                            rightPadding: 16
                                            color: "Orange"
                                        }
                                    }
                                }
                                ItemDelegate {
                                    Layout.fillWidth: true
                                    spacing: 0
                                    padding: 0

                                    contentItem: RowLayout {
                                        spacing: 0
                                        Layout.fillWidth: true
                                        Label {
                                            text: qsTr("Test %1c").arg(
                                                      view.currentIndex + 1)
                                            Layout.fillWidth: true
                                            leftPadding: 32
                                        }

                                        Label {
                                            text: "ERROR"
                                            rightPadding: 16
                                            color: "Red"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    PageIndicator {
        id: indicator

        count: view.count
        currentIndex: view.currentIndex

        anchors.bottom: view.bottom
        anchors.horizontalCenter: parent.horizontalCenter
    }
}
