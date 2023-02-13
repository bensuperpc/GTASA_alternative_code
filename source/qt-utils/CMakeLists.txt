##############################################################
#   ____                                                     #
#  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___    #
#  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __|   #
#  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__    #
#  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___|   #
#                             |_|             |_|            #
##############################################################
#                                                            #
#  BenLib, 2021                                              #
#  Created: 16, March, 2021                                  #
#  Modified: 17, March, 2021                                 #
#  file: CMakeLists.txt                                      #
#  CMake                                                     #
#  Source:                                                   #
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################

cmake_minimum_required(VERSION 3.14)
project(
    qt_utils_lib
    VERSION 0.1.0
    DESCRIPTION "utils library with Qt"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES C CXX
)

find_package(Qt6 6.4 COMPONENTS Quick Gui Core Qml Charts Multimedia REQUIRED)
qt_standard_project_setup()

set(SRCS
    imageprovider/imageprovider.cpp
    TableModel.cpp
	)
	
set(HEADERS
    imageprovider/imageprovider.h
    TableModel.h
    asyncimageprovider/asyncimageprovider.h
    asyncimageprovider/async_image_response_runnable.h
    asyncimageprovider/async_image_response.h
    )

add_library(qt_utils_lib ${SRCS} ${HEADERS})
target_include_directories(qt_utils_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qt_utils_lib PRIVATE Qt6::Quick Qt6::Gui Qt6::Core Qt6::Qml Qt6::Charts Qt6::Multimedia)
target_compile_features(qt_utils_lib PUBLIC cxx_std_20)

set_target_properties(qt_utils_lib
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
