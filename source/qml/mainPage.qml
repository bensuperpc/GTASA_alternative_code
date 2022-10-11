import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material

// import QtWebEngine
Page {
    title: qsTr("Main Page")
    id: page

    Flickable {
        anchors.fill: parent
        // contentHeight: columnLayout.implicitHeight
        // contentWidth: columnLayout.implicitWidth
        flickableDirection: Flickable.AutoFlickIfNeeded

        GridLayout {
            anchors.fill: parent
            columns: 5
            rows: 5

            component ProportionalRect: Rectangle {
                Layout.fillHeight: true
                Layout.fillWidth: true
                Layout.preferredWidth: Layout.columnSpan
                Layout.preferredHeight: Layout.rowSpan
            }

            ProportionalRect {
                Layout.columnSpan: 4
                Layout.rowSpan: 2

                color: "red"
            }

            ProportionalRect {
                Layout.columnSpan: 1
                Layout.rowSpan: 2

                color: "green"
            }

            ProportionalRect {
                Layout.columnSpan: 5
                Layout.rowSpan: 3

                color: "blue"


                /*
            WebEngineView {
                anchors.fill: parent
                url: "https: //www.qt.io"
            }*/
            }
        }


        /*
    ColumnLayout {
        // unique child
        id: columnLayout
        spacing: 10
        width: page.width // ensure correct width
        height: children.height // ensure correct height

        Label {
            Layout.alignment: Qt.AlignHCenter
            text: qsTr("You are on Main Page")
        }

        // your children hereon...
        Repeater {
            model: 4
            delegate: Rectangle {
            Layout.alignment: Qt.AlignHCenter
            width: 50
            height: 50
            color: "yellow"
        }
    }
}
*/
        ScrollIndicator.vertical: ScrollIndicator {}
        ScrollIndicator.horizontal: ScrollIndicator {}
        ScrollBar.vertical: ScrollBar {}
        ScrollBar.horizontal: ScrollBar {}
    }
}
