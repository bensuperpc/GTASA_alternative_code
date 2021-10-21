if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_INCLUDEDIR include/GTA_SA_cheat_finder CACHE PATH "")
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package GTA_SA_cheat_finder)

install(
    TARGETS GTA_SA_cheat_finder_GTA_SA_cheat_finder
    EXPORT GTA_SA_cheat_finderTargets
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
    FILES cmake/install-config.cmake
    DESTINATION "${GTA_SA_cheat_finder_INSTALL_CMAKEDIR}"
    RENAME "${package}Config.cmake"
    COMPONENT GTA_SA_cheat_finder_Development
)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${GTA_SA_cheat_finder_INSTALL_CMAKEDIR}"
    COMPONENT GTA_SA_cheat_finder_Development
)

install(
    EXPORT GTA_SA_cheat_finderTargets
    NAMESPACE GTA_SA_cheat_finder::
    DESTINATION "${GTA_SA_cheat_finder_INSTALL_CMAKEDIR}"
    COMPONENT GTA_SA_cheat_finder_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
