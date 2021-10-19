find_package(OpenMP)

if (OpenMP_FOUND OR OpenMP_CXX_FOUND)
    message(STATUS "OPENMP: FOUND")
    
    #link_libraries(OpenMP::OpenMP_CXX)

    target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    #set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(STATUS "OPENMP: NOT FOUND")
#    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()
