CC=g++
TARGET=gta
SOURCES=GTA_SA_cheat_finder.cxx
CFLAGS=-Ofast -march=native -fopenmp
LFLAGS=-fopenmp

OBJS=$(SOURCES:.cxx=.o)

all: $(TARGET)

%.o: %.cxx
	$(CC) $(CFLAGS) -c $<

$(TARGET): $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o $(TARGET)

purge: clean
	rm -f $(TARGET)

clean:
	rm -f *.o
