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
#//  Modified: 31, June, 2021                                //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

CXX=g++
TARGET=gta
SOURCES=src/GTA_SA_cheat_finder.cpp
HEADERS=src/GTA_SA_cheat_finder.hpp
CFLAGS=-O3 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fopenmp
LFLAGS=-fopenmp

OBJS=$(SOURCES:.cpp=.o)

all: $(TARGET)

%.o: %.cpp Makefile
	$(CXX) $(CFLAGS) -c -o $@ $<

$(TARGET): $(OBJS) $(HEADERS)
	$(CXX) $(LFLAGS) $(OBJS) -o $(TARGET)

cmake:
	cmake -Bbuild-local -H. -GNinja -DCMAKE_BUILD_TYPE=Release && ninja -Cbuild-local

docker:
	docker build . -t bensuperpc/gta:latest

push: docker
	docker push bensuperpc/gta:latest

purge: clean
	@rm -f $(TARGET)
	@rm -rf build-*
	@rm -rf dockcross-*

clean:
	@rm -f $(OBJS)

.PHONY: clean purge docker cmake