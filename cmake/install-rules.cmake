install(
    TARGETS GTASA_alternative_code_exe
    RUNTIME COMPONENT GTASA_alternative_code_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
