import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Window

import org.bensuperpc.application 1.0

import AppLib 1.0

Page {
    title: qsTr("About")
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
                HorizontalLabelDelegateMenu {
                    leftText: "App version"
                    rightText: AppSingleton.appVersion()
                }
                HorizontalLabelDelegateMenu {
                    leftText: "Author(s)"
                    rightText: AppSingleton.appAuthor()
                }
                HorizontalLabelDelegateMenu {
                    leftText: "Copyright"
                    rightText: AppSingleton.appBusiness()
                }
                CustomMenuSeparator {
                    color: window.Material.dividerColor
                }
                HorizontalLabelDelegateMenu {
                    leftText: "Build date"
                    rightText: AppSingleton.buildDate()
                }
                HorizontalLabelDelegateMenu {
                    leftText: "Build type"
                    rightText: AppSingleton.buildType()
                }
                CustomMenuSeparator {
                    color: window.Material.dividerColor
                }
                HorizontalLabelDelegateMenu {
                    leftText: "OS"
                    rightText: AppSingleton.osType()
                }
                HorizontalLabelDelegateMenu {
                    leftText: "OS version"
                    rightText: AppSingleton.osTypeFull()
                }
                CustomMenuSeparator {
                    color: window.Material.dividerColor
                }
                HorizontalLabelDelegateMenu {
                    leftText: "Compiler"
                    rightText: AppSingleton.compilerName()
                }
                HorizontalLabelDelegateMenu {
                    leftText: "Compiler arch"
                    rightText: AppSingleton.compilerArch()
                }
                HorizontalLabelDelegateMenu {
                    leftText: "Compiler version"
                    rightText: AppSingleton.compilerVersion()
                }
                CustomMenuSeparator {
                    color: window.Material.dividerColor
                }
                HorizontalLabelDelegateMenu {
                    leftText: "C++ version"
                    rightText: AppSingleton.cxxVersion()
                }
                CustomMenuSeparator {
                    color: window.Material.dividerColor
                }
                HorizontalLabelDelegateMenu {
                    leftText: "C version"
                    rightText: AppSingleton.cVersion()
                }
                CustomMenuSeparator {
                    color: window.Material.dividerColor
                }
                HorizontalLabelDelegateMenu {
                    leftText: "Qt version"
                    rightText: AppSingleton.qtVersion()
                }
                VerticalLabelDelegateMenu {
                    upText: "Qt version"
                    downText: AppSingleton.qtVersion()
                }
            }
        }
    }
}
