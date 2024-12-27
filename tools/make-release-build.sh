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
find . -maxdepth 1 -type f -name 'GTASA_alternative_code*' ! -name '*.cmake' | xargs sha256sum >checksum.sha256 && echo "checksum: OK"
{ find . ! -name . \( -name 'GTASA_alternative_code*' -o -name '*.sha256' \) ! \( -name '*.*' -o -name '*.*' \) && find .. -maxdepth 1 -type f -name '*.md' && find .. -maxdepth 1 -type f -name '*.sha256'; } | XZ_OPT=-e9 tar -cJf ../gta-local.tar.xz -T -
