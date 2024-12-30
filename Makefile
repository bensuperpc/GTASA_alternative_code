#//////////////////////////////////////////////////////////////
#//                                                          //
#//  GTASA_alternative_code, 2023                               //
#//  Created: 04, June, 2021                                 //
#//  Modified: 18, November, 2023                            //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

PROJECT_NAME := GTASA_alternative_code

PROJECT_ROOT := .

# -Werror=float-equal
#-DRUN_HAVE_STD_REGEX=1 -DHAVE_STD_REGEX=ON
CMAKE_ADDITONAL_ARGS ?= -DENABLE_OPENCL=OFF -DENABLE_CUDA=ON -DENABLE_OPENMP=ON -DENABLE_QT=OFF -DENABLE_THREADPOOL=ON

include CPPProject.mk
