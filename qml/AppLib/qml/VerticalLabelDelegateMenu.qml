import QtQuick
import QtQuick.Layouts
import QtQuick.Controls

ItemDelegate {
    Layout.fillWidth: true
    spacing: 0
    padding: 0
    property string upText: ""
    property string downText: ""

    contentItem: ColumnLayout {
        spacing: 0
        Layout.fillWidth: true
        // Up left text
        Label {
            text: upText
            Layout.fillWidth: true
            leftPadding: 48
        }

        // Down left text
        Label {
            text: downText
            Layout.fillWidth: true
            leftPadding: 48
            font.pixelSize: 12
            font.weight: Font.Light
        }

        // Middle right text
        /*
        Label {
            text: "Right"
            Layout.alignment: Qt.AlignRight
            // Layout.verticalAlignment: Qt.AlignVCenter
            Layout.fillWidth: true
            rightPadding: 32
        }
        */
    }
}
