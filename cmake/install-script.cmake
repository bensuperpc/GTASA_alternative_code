file(
    RELATIVE_PATH relative_path
    "/${GTA_SA_cheat_finder_INSTALL_CMAKEDIR}"
    "/${CMAKE_INSTALL_BINDIR}/${GTA_SA_cheat_finder_NAME}"
)

get_filename_component(prefix "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
set(config_dir "${prefix}/${GTA_SA_cheat_finder_INSTALL_CMAKEDIR}")
set(config_file "${config_dir}/GTA_SA_cheat_finderConfig.cmake")

message(STATUS "Installing: ${config_file}")
file(WRITE "${config_file}" "\
get_filename_component(
    _GTA_SA_cheat_finder_executable
    \"\${CMAKE_CURRENT_LIST_DIR}/${relative_path}\"
    ABSOLUTE
)
set(
    GTA_SA_CHEAT_FINDER_EXECUTABLE \"\${_GTA_SA_cheat_finder_executable}\"
    CACHE FILEPATH \"Path to the GTA_SA_cheat_finder executable\"
)
")
list(APPEND CMAKE_INSTALL_MANIFEST_FILES "${config_file}")
