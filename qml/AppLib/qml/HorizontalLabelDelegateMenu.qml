import QtQuick
import QtQuick.Layouts
import QtQuick.Controls

ItemDelegate {
    Layout.fillWidth: true
    spacing: 0
    padding: 0
    property string leftText: ""
    property string rightText: ""

    contentItem: RowLayout {
        spacing: 0
        Layout.fillWidth: true
        Label {
            text: leftText
            Layout.fillWidth: true
            leftPadding: 48
        }

        Label {
            text: rightText
            rightPadding: 24
        }
    }
}
