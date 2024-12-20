cmake_minimum_required(VERSION 3.22.0)

project(
    AppLib
    VERSION 0.0.1
    DESCRIPTION ""
    HOMEPAGE_URL "https://www.bensuperpc.org"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_DEBUG_QT "Enable debug messages for Qt" OFF)
set(QT_VERSION_MAJOR 6 CACHE BOOL "" FORCE)
set(QT_VERSION_MINOR 8 CACHE BOOL "" FORCE)

find_package(QT NAMES Qt6 COMPONENTS Quick QuickControls2 Gui Core Qml Charts Multimedia Concurrent VirtualKeyboard REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} ${QT_VERSION_MAJOR}.${QT_VERSION_MINOR} COMPONENTS Quick QuickControls2 Gui Core 
    3DCore Qml Charts Multimedia Concurrent VirtualKeyboard)

qt_standard_project_setup(REQUIRES ${QT_VERSION_MAJOR}.${QT_VERSION_MINOR})

set(QML_PLUGIN_FILES
    qml/ProportionalRect.qml
    qml/CustomMenuSeparator.qml
    qml/HorizontalLabelDelegateMenu.qml
    qml/VerticalLabelDelegateMenu.qml
    )

# Standard library
set(LIB_SOURCES
    )

set(LIB_HEADER
    )

set(QML_PLUGIN_RESOURCES
    )

set(QML_PLUGIN_IMPORT_PATH
    qml
    source/applib/qml

    source
    )

include_directories(${QML_PLUGIN_IMPORT_PATH})

qt_add_library(${PROJECT_NAME} STATIC)
qt_add_qml_module(
    ${PROJECT_NAME}
    URI ${PROJECT_NAME}
    PLUGIN_TARGET ${PROJECT_NAME}plugin
    NO_PLUGIN_OPTIONAL
    NO_GENERATE_PLUGIN_SOURCE
    CLASS_NAME AppLibPlugin
    OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    RESOURCE_PREFIX "/bensuperpc.org"
    IMPORT_PATH ${QML_PLUGIN_IMPORT_PATH}
    QML_IMPORT_PATH ${QML_PLUGIN_IMPORT_PATH}
    VERSION 1.0
    QML_FILES
    ${QML_PLUGIN_FILES}
    SOURCES
    ${SOURCES}
    ${HEADER}
    RESOURCES
    ${QML_PLUGIN_RESOURCES}
)

target_sources(${PROJECT_NAME}plugin PRIVATE
    source/applib/plugin/AppLibplugin_AppLibPlugin.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${PROJECT_NAME} PUBLIC source)
target_include_directories(${PROJECT_NAME} PUBLIC source/applib)
target_include_directories(${PROJECT_NAME}plugin PUBLIC source)
target_include_directories(${PROJECT_NAME}plugin PUBLIC source/applib)
target_include_directories(${PROJECT_NAME}plugin PUBLIC source/applib/qml)

if(${PROJECT_NAME})
    target_compile_definitions(${PROJECT_NAME} PUBLIC QT_MESSAGELOGCONTEXT="1")
    target_compile_definitions(${PROJECT_NAME}plugin PUBLIC QT_MESSAGELOGCONTEXT="1")
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC Qt${QT_VERSION_MAJOR}::Quick
    Qt${QT_VERSION_MAJOR}::QuickControls2
    Qt${QT_VERSION_MAJOR}::Charts Qt${QT_VERSION_MAJOR}::Concurrent
    Qt${QT_VERSION_MAJOR}::3DCore Qt${QT_VERSION_MAJOR}::VirtualKeyboard)

target_link_libraries(${PROJECT_NAME}plugin PUBLIC Qt${QT_VERSION_MAJOR}::Quick
    Qt${QT_VERSION_MAJOR}::QuickControls2
    Qt${QT_VERSION_MAJOR}::Charts Qt${QT_VERSION_MAJOR}::Concurrent
    Qt${QT_VERSION_MAJOR}::3DCore Qt${QT_VERSION_MAJOR}::VirtualKeyboard)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(${PROJECT_NAME}plugin
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
