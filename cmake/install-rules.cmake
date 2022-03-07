include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package GTA_SA_cheat_finder)

install(
    TARGETS GTA_SA_cheat_finder_exe
    RUNTIME COMPONENT GTA_SA_cheat_finder_Runtime
)

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

# Allow package maintainers to freely override the path for the configs
set(
    GTA_SA_cheat_finder_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/${package}"
    CACHE PATH "CMake package config location relative to the install prefix"
)
mark_as_advanced(GTA_SA_cheat_finder_INSTALL_CMAKEDIR)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${GTA_SA_cheat_finder_INSTALL_CMAKEDIR}"
    COMPONENT GTA_SA_cheat_finder_Development
)

# Export variables for the install script to use
install(CODE "
set(GTA_SA_cheat_finder_NAME [[$<TARGET_FILE_NAME:GTA_SA_cheat_finder_exe>]])
set(GTA_SA_cheat_finder_INSTALL_CMAKEDIR [[${GTA_SA_cheat_finder_INSTALL_CMAKEDIR}]])
set(CMAKE_INSTALL_BINDIR [[${CMAKE_INSTALL_BINDIR}]])
" COMPONENT GTA_SA_cheat_finder_Development)

install(
    SCRIPT cmake/install-script.cmake
    COMPONENT GTA_SA_cheat_finder_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
