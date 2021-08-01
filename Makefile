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
#//  Created: 14, June, 2021                                 //
#//  Modified: 1, August, 2021                               //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

DOCKER_IMAGE = linux-s390x android-arm android-arm64 linux-x86 linux-x64 linux-mips linux-x64-clang manylinux1-x64 manylinux1-x86 manylinux2010-x64 manylinux2010-x86 manylinux2014-x64 manylinux2014-x86 manylinux2014-aarch64 web-wasm linux-arm64 linux-arm64-musl windows-static-x86 windows-static-x64 windows-static-x64-posix windows-shared-x86 windows-shared-x64 windows-shared-x64-posix linux-armv7 linux-armv7a linux-armv7l-musl linux-armv6 linux-armv6-musl linux-armv5 linux-armv5-musl linux-ppc64le linux-riscv64 linux-riscv32 linux-m68k-uclibc

default: linux-x64


all: $(DOCKER_IMAGE)


$(DOCKER_IMAGE):
	./builder/dockcross-builder.sh $@

ninja:
	cmake -Bbuild-local -H. -GNinja -DCMAKE_BUILD_TYPE=Release && ninja -Cbuild-local

docker:
	docker build . -t bensuperpc/gta:latest

push: docker
	docker push bensuperpc/gta:latest

clean:
	@rm -rf build-*
	@rm -rf dockcross-*

.PHONY: clean purge docker ninja $(DOCKER_IMAGE)
