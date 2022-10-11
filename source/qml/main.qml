import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Window

ApplicationWindow {
    property bool isOpened: false
    property bool inPortrait: window.width < window.height

    //Material.theme: Material.Dark
    //Material.primary: Material.Amber
    Material.theme: "Dark"
    Material.primary: "Amber"
    Material.accent: "Teal"

    id: window

    visible: true
    width: 720
    height: 1080

    title: qsTr("KrackX")

    header: ToolBar {
        id: toolbar
        RowLayout {
            anchors.fill: parent
            ToolButton {
                id: toolButton1
                text: {
                    qsTr(stackView.depth > 1 ? "<" : "\u2630")
                }
                font.bold: true
                antialiasing: true
                onClicked: {
                    if (stackView.depth > 1) {
                        stackView.pop()
                    } else {
                        drawer.open()
                    }
                }
            }
            Label {
                id: title_label
                text: stackView.currentItem.title
                elide: Label.ElideRight
                horizontalAlignment: Qt.AlignHCenter
                verticalAlignment: Qt.AlignVCenter
                Layout.fillWidth: true
            }
            ToolButton {
                text: qsTr("⋮")
                font.bold: true
                antialiasing: true
                onClicked: menu.open()
            }
        }
    }

    Menu {
        id: menu
        x: window.width
        width: window.width * 0.37
        // transformOrigin: Menu.TopRight
        MenuItem {
            id: parametres
            text: "Paramètres"
            onTriggered: {
                console.log("onTriggered " + parametres.text)
                stackView.push("SettingsPage.qml")
            }
        }
        MenuItem {
            id: quit
            text: "Quit"
            onTriggered: {
                console.log("onTriggered " + quit.text)
                Qt.quit()
            }
        }
        MenuItem {
            width: parent.width
            height: children[0].height
            enabled: false
            MenuSeparator {
                width: parent.width
            }
        }

        MenuItem {
            id: help
            text: "Help"
            onTriggered: {
                console.log("onTriggered " + help.text)
                aPropos.open()
            }
        }
        MenuItem {
            id: about
            text: "About"
            onTriggered: {
                console.log("onTriggered " + about.text)
                stackView.push("AboutPage.qml")
            }
        }
    }


    /*
    Dialog {
        id: aPropos
        modal: true
        focus: true
        title: "About"
        x: (window.width - width) / 2
        y: window.height / 6
        width: Math.min(window.width, window.height) / 3 * 2
        contentHeight: message.height
        Label {
            id: message
            width: aPropos.availableWidth
            text: "Application built with Qt quick."
            wrapMode: Label.Wrap
            font.pixelSize: 12
        }
    }
*/
    Dialog {
        id: aPropos
        focus: true
        title: "About"
        anchors.centerIn: parent
        Text {
            id: field
            width: parent.width
            height: parent.height
            text: "Application built with Qt quick."
            minimumPointSize: 5
            font.pointSize: 25
            fontSizeMode: Text.Fit
            color: 'white'
            anchors.centerIn: parent
        }
    }

    Drawer {
        id: drawer

        //In portrait mode
        //y: toolbar.height

        //!inPortrait ? idContentColumn.height : idContentColumn.width
        //y: inPortrait ? toolbar.height : toolbar.width

        // width: window.width * 0.66
        // height: window.height

        //dragMargin: window.height * 0.1
        width: window.width * 0.6
        height: window.height

        // dragMargin: window.height * 0.03
        onOpened: {
            console.log("drawer onOpened")
            isOpened = true
            myApp.quitSignalSlot()
            myApp.quitSignalInvokable()
        }
        onClosed: {
            console.log("drawer onClosed")
            isOpened = false
        }

        Flickable {
            //Fix issue with wrong Flickable size in !inPortrait
            contentHeight: !inPortrait ? idContentColumn.height : idContentColumn.width
            anchors.fill: parent
            clip: true
            Column {
                width: parent.width
                height: parent.height
                id: idContentColumn
                ItemDelegate {
                    text: qsTr("MainPage")
                    width: parent.width
                    onClicked: {
                        if (stackView.depth > 1) {
                            stackView.push("mainPage.qml")
                            drawer.close()
                        } else {
                            drawer.close()
                            // To remove later
                            stackView.push("mainPage.qml")
                        }
                    }
                }
                ItemDelegate {
                    width: parent.width
                    height: menu_separator.height
                    anchors.horizontalCenter: parent.horizontalCenter
                    enabled: false
                    MenuSeparator {
                        id: menu_separator
                        width: parent.width
                        anchors.horizontalCenter: parent.horizontalCenter
                    }
                }
                ItemDelegate {
                    text: qsTr("Settings")
                    width: parent.width // toute la largeur du tiroir
                    onClicked: {
                        stackView.push("SettingsPage.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }
                ItemDelegate {
                    text: qsTr("KrackPasswordPage")
                    width: parent.width
                    onClicked: {
                        stackView.push("KrackPasswordPage.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }
                ItemDelegate {
                    text: qsTr("GTA_SA")
                    width: parent.width
                    onClicked: {
                        stackView.push("GTA_SA.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }
                ItemDelegate {
                    width: parent.width
                    height: menu_separator1.height
                    anchors.horizontalCenter: parent.horizontalCenter
                    enabled: false
                    MenuSeparator {
                        id: menu_separator1
                        width: parent.width
                        anchors.horizontalCenter: parent.horizontalCenter
                    }
                }
                ItemDelegate {
                    text: qsTr("Image_test")
                    width: parent.width // toute la largeur du tiroir
                    onClicked: {
                        stackView.push("image_test.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }
                ItemDelegate {
                    width: parent.width
                    height: menu_separator3.height
                    anchors.horizontalCenter: parent.horizontalCenter
                    enabled: false
                    MenuSeparator {
                        id: menu_separator3
                        width: parent.width
                        anchors.horizontalCenter: parent.horizontalCenter
                    }
                }
                ItemDelegate {
                    id: choix4
                    text: qsTr("About")
                    width: parent.width
                    onClicked: {
                        console.log("onClicked " + choix4.text)
                        stackView.push("AboutPage.qml")
                        drawer.close() // et on referme le tiroir
                    }
                }
            }
        }
    }

    StackView {
        id: stackView
        initialItem: "GTA_SA.qml"
        anchors.fill: parent
        width: parent.width
        height: parent.height

        // anchors.centerIn: parent
        pushEnter: Transition {
            PropertyAnimation {
                property: "opacity"
                from: 0
                to: 1
                duration: 200
            }
        }
        pushExit: Transition {
            PropertyAnimation {
                property: "opacity"
                from: 1
                to: 0
                duration: 200
            }
        }
        popEnter: Transition {
            PropertyAnimation {
                property: "opacity"
                from: 0
                to: 1
                duration: 200
            }
        }
        popExit: Transition {
            PropertyAnimation {
                property: "opacity"
                from: 1
                to: 0
                duration: 200
            }
        }
    }

    Component.onDestruction: {
        console.log("Closing app...")
    }
}
