FROM alpine:latest as builder
RUN apk add --no-cache gcc g++ ninja cmake
WORKDIR /app
COPY src/GTA_SA_cheat_finder.cpp src/GTA_SA_cheat_finder.cpp
COPY src/GTA_SA_cheat_finder.hpp src/GTA_SA_cheat_finder.hpp
COPY CMakeLists.txt .
RUN cmake -Bbuild -H. -GNinja -DCMAKE_BUILD_TYPE=Release && ninja -Cbuild

FROM alpine:latest as runtime
RUN apk add --no-cache libstdc++ libgomp libgcc
COPY --from=builder app/build/gta .
# Test exec
RUN ./gta 0 52
ENTRYPOINT ["./gta"] 
