#//////////////////////////////////////////////////////////////
#//                                                          //
#//  GTA_SA_cheat_finder, 2023                               //
#//  Created: 04, June, 2021                                 //
#//  Modified: 18, November, 2023                            //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

PROJECT_NAME := GTA_SA_cheat_finder

PROJECT_ROOT := .

#-DRUN_HAVE_STD_REGEX=1
CMAKE_ADDITONAL_ARGS := -DHAVE_STD_REGEX=ON

include CPPProject.mk
