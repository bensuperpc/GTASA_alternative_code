cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    GTA_SA_cheat_finder
    VERSION 2.0.0
    DESCRIPTION "Find alternative cheat code in Grand Theft Auto San Andreas (2004) by bruteforce"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES C CXX
)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
endif()

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Enable Utile ----
include(cmake/utile/ccache.cmake)
# include(cmake/utile/ninja_color.cmake)

find_package(QT NAMES Qt6 COMPONENTS Quick QuickControls2 Gui Core Qml Charts Multimedia Widgets Concurrent SerialPort)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Quick QuickControls2 Gui Core Qml Charts Multimedia Widgets Concurrent SerialPort)

if (Qt${QT_VERSION_MAJOR}_FOUND)
    if (Qt${QT_VERSION_MAJOR}_VERSION VERSION_GREATER_EQUAL 6.5)
        qt_standard_project_setup()
    else()
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTORCC ON)
        set(CMAKE_AUTOUIC ON)
        set(CMAKE_AUTOSRC ON)
    endif()
else()
    message(WARNING "Qt${QT_VERSION_MAJOR} not found, project will be CLI only")
endif()

find_package(OpenMP 2.0)

find_package(CUDAToolkit 11.0)

find_package(OpenCL 2.0)

# Download ThreadPool
include(cmake/lib/threadpool.cmake)

add_subdirectory(source/gta_cheat_finder)

# ---- Declare executable ----

set(SRCS
    source/main.cpp
	)
	
set(HEADERS
    )

if (Qt${QT_VERSION_MAJOR}_FOUND)
    list(APPEND SRCS 
        source/gta_sa_ui.cpp
        source/imageprovider.cpp
        source/application.cpp
        source/tablemodel.cpp
        )
    list(APPEND HEADERS 
        source/gta_sa_ui.hpp
        source/asyncimageprovider.h
        source/application.h 
        source/tablemodel.h 
        source/utils/about_compilation.h
        source/utils/utils.h
        )
endif()

if (Qt${QT_VERSION_MAJOR}_FOUND)
    qt_add_executable(GTA_SA_cheat_finder_exe
        ${SRCS} ${HEADERS}
    )
else()
    add_executable(GTA_SA_cheat_finder_exe
        ${SRCS} ${HEADERS}
    )
endif()

add_executable(GTA_SA_cheat_finder::exe ALIAS GTA_SA_cheat_finder_exe)


if (Qt${QT_VERSION_MAJOR}_FOUND)
    qt_add_qml_module(GTA_SA_cheat_finder_exe
        URI bensuperpc
        OUTPUT_DIRECTORY bensuperpc
        RESOURCE_PREFIX /bensuperpc.org
        VERSION 1.0
        QML_FILES
        qml/responsive.qml
        qml/main.qml
        qml/home.qml qml/image.qml
        qml/table.qml
        qml/settings.qml
        qml/swipe.qml
        qml/custom/ProportionalRect.qml
        qml/custom/CustomMenuSeparator.qml
        qml/custom/HorizontalLabelDelegateMenu.qml
        qml/custom/VerticalLabelDelegateMenu.qml
        IMPORT_PATH qml
        RESOURCES
        image/cat_sticking_out_its_tongue.jpg
    )
endif()

set_target_properties(GTA_SA_cheat_finder_exe PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

target_compile_features(GTA_SA_cheat_finder_exe PUBLIC cxx_std_20)

if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
  target_link_libraries(GTA_SA_cheat_finder_exe PUBLIC OpenMP::OpenMP_CXX)
endif()


if (CUDAToolkit_FOUND)
  target_include_directories(GTA_SA_cheat_finder_exe PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

target_link_libraries(GTA_SA_cheat_finder_exe PUBLIC GTA_SA_cheat_finder_lib)

set_property(TARGET GTA_SA_cheat_finder_exe PROPERTY OUTPUT_NAME GTA_SA_cheat_finder)

target_compile_features(GTA_SA_cheat_finder_exe PRIVATE cxx_std_20)

if (Qt${QT_VERSION_MAJOR}_FOUND)
    target_link_libraries(GTA_SA_cheat_finder_exe
        PRIVATE Qt${QT_VERSION_MAJOR}::Quick Qt${QT_VERSION_MAJOR}::QuickControls2 Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::Charts Qt${QT_VERSION_MAJOR}::Concurrent Qt${QT_VERSION_MAJOR}::SerialPort)
endif()

target_compile_definitions(GTA_SA_cheat_finder_exe PUBLIC PROJECT_VERSION="${PROJECT_VERSION}")
target_compile_definitions(GTA_SA_cheat_finder_exe PUBLIC CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
target_compile_definitions(GTA_SA_cheat_finder_exe PUBLIC CMAKE_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}")
target_compile_definitions(GTA_SA_cheat_finder_exe PUBLIC CMAKE_C_COMPILER_ID="${CMAKE_C_COMPILER_ID}")
target_compile_definitions(GTA_SA_cheat_finder_exe PUBLIC CMAKE_VERSION="${CMAKE_VERSION}")

configure_file(image/cat_sticking_out_its_tongue.jpg ${CMAKE_BINARY_DIR}/image/cat_sticking_out_its_tongue.jpg COPYONLY)


# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT GTA_SA_cheat_finder_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of GTA_SA_cheat_finder"
  )
endif()

include(cmake/dev-mode.cmake)
