import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Window

import org.bensuperpc.application 1.0

import "custom/"

Page {
    title: qsTr("Settings")
    id: page

    Flickable {
        anchors.fill: parent
        contentWidth: width
        contentHeight: flow.implicitHeight
        Flow {
            id: flow
            width: parent.width
            spacing: 0

            ColumnLayout {
                width: parent.width
                clip: true
                spacing: 0

                RowLayout {
                    Layout.fillWidth: true
                    SwitchDelegate {
                        Layout.fillWidth: true
                        leftPadding: 48
                        rightPadding: 32
                        text: "Dark Theme"
                        checked: window.Material.theme === Material.Dark ? true : false

                        onClicked: {
                            window.Material.theme = this.checked ? Material.Dark : Material.Light
                        }
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    SwitchDelegate {
                        Layout.fillWidth: true
                        leftPadding: 48
                        rightPadding: 32
                        text: "Color primary"
                        checked: window.Material.primary === Material.color(
                                     Material.BlueGrey) ? true : false

                        onClicked: {
                            window.Material.primary
                                    = this.checked ? Material.BlueGrey : Material.Indigo
                        }
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    SwitchDelegate {
                        Layout.fillWidth: true
                        leftPadding: 48
                        rightPadding: 32
                        text: "Color accent"
                        checked: true
                        onClicked: {
                            window.Material.accent = this.checked ? Material.Teal : Material.Orange
                        }
                    }
                }

                CustomMenuSeparator {
                    color: window.Material.dividerColor
                }
                RowLayout {
                    Layout.fillWidth: true
                    SwitchDelegate {
                        Layout.fillWidth: true
                        leftPadding: 48
                        rightPadding: 32
                        text: "Debug mode"
                        checked: false
                        enabled: false
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 0
                    enabled: false
                    ButtonGroup {
                        id: buttonGroupLanguage
                    }
                    ItemDelegate {
                        Layout.fillWidth: true
                        leftPadding: 48
                        text: "Language"
                    }

                    RadioDelegate {
                        checked: true
                        text: "en"
                        ButtonGroup.group: buttonGroupLanguage
                    }
                    RadioDelegate {
                        checked: true
                        text: "es"
                        ButtonGroup.group: buttonGroupLanguage
                    }
                    RadioDelegate {
                        rightPadding: 24
                        text: "fr"
                        ButtonGroup.group: buttonGroupLanguage
                    }
                }
                CustomMenuSeparator {
                    color: window.Material.dividerColor
                }
            }
        }
    }
}
