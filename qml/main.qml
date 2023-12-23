import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Window

ApplicationWindow {
    width: 720
    height: 1280

    minimumWidth: 400
    minimumHeight: 480

    visible: true
    title: qsTr("main")

    id: window

    // Default theme
    Material.theme: Material.Dark
    Material.primary: Material.BlueGrey
    Material.accent: Material.Teal

    //Material.foreground: "White"
    //Material.background: "Grey"
    //property bool inPortrait: window.width < window.height

    // List all pages and names
    property var drawerPageFile: ["responsive.qml", "home.qml", "graphic.qml", "table.qml", "image.qml", "swipe.qml", "settings.qml"]
    property var drawerPageName: ["GTA_SA", "Home", "Graphic", "Table", "Image", "Swipe", "Settings"]
    property int currentPageIndex: 0

    header: ToolBar {
        id: toolbar
        RowLayout {
            anchors.fill: parent
            ToolButton {
                id: toolButton1
                text: qsTr("=")
                font.pixelSize: 22
                font.bold: true
                onClicked: {
                    drawer.open()
                }
            }
            Label {
                id: title_label
                text: stackView.currentItem.title
                font.bold: true
                horizontalAlignment: Qt.AlignHCenter
                verticalAlignment: Qt.AlignVCenter
                Layout.fillWidth: true
                clip: true
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
            text: qsTr("Quit")
            onTriggered: {
                Qt.quit()
            }
        }
        MenuSeparator {}

        MenuItem {
            text: qsTr("Help")
            onTriggered: {
                aPropos.open()
            }
        }
        MenuItem {
            text: qsTr("About")
            onTriggered: {
                currentPageIndex = drawerPageFile.length + 1
                stackView.push("about.qml")
            }
        }
    }

    Dialog {
        id: aPropos
        focus: true
        title: "About"
        x: (parent.width - width) / 2
        y: (parent.height - height) / 2
        Label {
            id: message
            width: aPropos.availableWidth
            text: qsTr("Application built with Qt quick.")
            wrapMode: Label.Wrap
            font.pixelSize: 12
        }
    }

    Drawer {
        id: drawer
        width: window.width * 0.75
        height: window.height

        dragMargin: window.height * 0.03
        onOpened: {
            console.log("drawer onOpened")
        }
        onClosed: {
            console.log("drawer onClosed")
        }

        ListView {
            id: listView
            anchors.fill: parent

            headerPositioning: ListView.OverlayHeader
            header: Pane {
                id: header
                z: 2
                width: listView.width
                padding: 6
                clip: true

                //contentHeight: logo.height
                RowLayout {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    Image {
                        id: logo
                        source: Qt.resolvedUrl(
                                    "/bensuperpc.org/bensuperpc/image/cat_sticking_out_its_tongue.jpg")
                        fillMode: Image.Stretch
                        Layout.preferredWidth: 64
                        Layout.preferredHeight: 64
                        asynchronous: true
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        clip: true
                        Label {
                            id: labelUserName
                            text: qsTr("Luz")
                            horizontalAlignment: Qt.AlignHCenter
                            //verticalAlignment: Qt.AlignVCenter
                            Layout.fillWidth: true
                            font.bold: true
                        }
                        Label {
                            id: labelUserLastName
                            text: qsTr("Noceda")
                            horizontalAlignment: Qt.AlignHCenter
                            Layout.fillWidth: true
                            font.pixelSize: 15
                            font.weight: Font.Light
                        }
                    }
                    ComboBox {
                        model: ["Luz", "Amity", "Willow", "Gus", "Hunter"]
                        implicitContentWidthPolicy: ComboBox.WidestText

                        onActivated: {
                            switch (currentIndex) {
                            case 0:
                                labelUserName.text = "Luz"
                                labelUserLastName.text = "Noceda"
                                break
                            case 1:
                                labelUserName.text = "Amity"
                                labelUserLastName.text = "Blight"
                                break
                            case 2:
                                labelUserName.text = "Willow"
                                labelUserLastName.text = "Park"
                                break
                            case 3:
                                labelUserName.text = "Gus"
                                labelUserLastName.text = "Porter"
                                break
                            case 4:
                                labelUserName.text = "Hunter"
                                labelUserLastName.text = "Gray"
                                break
                            default:
                                console.warn("User not found!")
                                break
                            }
                        }
                    }
                }
                MenuSeparator {
                    parent: header
                    width: listView.width
                    anchors.verticalCenter: parent.bottom
                }
            }

            footer: ItemDelegate {
                id: footer
                text: qsTr("About")
                font.bold: true
                width: listView.width

                onClicked: {
                    currentPageIndex = drawerPageFile.length + 1
                    stackView.push("about.qml")
                    drawer.close()
                }

                MenuSeparator {
                    parent: footer
                    width: listView.width
                    anchors.verticalCenter: parent.top
                }
            }

            model: drawerPageFile.length
            delegate: ItemDelegate {
                text: drawerPageName[index]
                width: listView.width

                highlighted: currentPageIndex === index ? true : false
                font.bold: index === 0 ? true : false

                onClicked: {
                    if (currentPageIndex !== index) {
                        currentPageIndex = index
                        stackView.push(drawerPageFile[index])
                    }
                    drawer.close()
                }
            }

            ScrollIndicator.vertical: ScrollIndicator {}
        }
    }

    StackView {
        id: stackView
        initialItem: drawerPageFile[currentPageIndex]
        anchors.fill: parent

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

    Component.onCompleted: {

    }

    Component.onDestruction: {

    }
}
