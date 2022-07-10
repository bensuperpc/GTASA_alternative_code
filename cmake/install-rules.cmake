install(
    TARGETS GTA_SA_cheat_finder_exe
    RUNTIME COMPONENT GTA_SA_cheat_finder_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
