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

CC=g++
TARGET=gta
SOURCES=GTA_SA_cheat_finder.cpp
HEADERS=GTA_SA_cheat_finder.hpp
CFLAGS=-O3 -march=native -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fopenmp #-std=c++17
LFLAGS=-fopenmp

OBJS=$(SOURCES:.cpp=.o)

all: $(TARGET)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $<

$(TARGET): $(OBJS) $(HEADERS)
	$(CC) $(LFLAGS) $(OBJS) -o $(TARGET)

purge: clean
	@rm -f $(TARGET)

clean:
	@rm -f $(OBJS)
