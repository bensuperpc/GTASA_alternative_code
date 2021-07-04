FROM alpine:latest as builder
RUN apk add --no-cache gcc g++ make
COPY Makefile .
COPY src/GTA_SA_cheat_finder.cpp src/GTA_SA_cheat_finder.cpp
COPY src/GTA_SA_cheat_finder.hpp src/GTA_SA_cheat_finder.hpp
RUN make && make clean

FROM alpine:latest as runtime
RUN apk add --no-cache libstdc++ libgomp libgcc
COPY --from=builder gta .
ENTRYPOINT ["./gta"] 
