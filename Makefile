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
#-DRUN_HAVE_STD_REGEX=1
CMAKE_ADDITONAL_ARGS := -DHAVE_STD_REGEX=ON

include CPPProject.mk
