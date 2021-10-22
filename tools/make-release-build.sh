#!/bin/bash

#//////////////////////////////////////////////////////////////
#//   ____                                                   //
#//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
#//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
#//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
#//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
#//                             |_|             |_|          //
#//////////////////////////////////////////////////////////////
#//                                                          //
#//  Script, 2021                                            //
#//  Modified: 22, October, 2021                             //
#//  Modified: 22, October, 2021                             //
#//  file: -                                                 //
#//  -                                                       //
#//  Source: https://github.com/bensuperpc/scripts                                                //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

cd build
find . -maxdepth 1 -type f -name 'GTA_SA_cheat_finder*' ! -name '*.cmake' | xargs sha256sum > checksum.sha256 && echo "checksum: OK"
{ find . -maxdepth 1 ! -name . \( -name 'GTA_SA_cheat_finder*' -o -name '*.sha256' \)  ! \( -name '*.cmake' -o -name '*.o' -o -name '*.gdb' \) && find .. -maxdepth 1 -type f -name '*.md' ; } | XZ_OPT=-e9 tar -cJf ../gta-local.tar.xz -T -
