install(
    TARGETS GTASA_alternative_code_exe
    RUNTIME COMPONENT GTASA_alternative_code_Runtime
    BUNDLE DESTINATION bin
#    LIBRARY DESTINATION lib${LIB_SUFFIX}
#    ARCHIVE DESTINATION lib${LIB_SUFFIX}
#    RUNTIME DESTINATION bin
#    BUNDLE DESTINATION bin
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
