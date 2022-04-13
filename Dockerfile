FROM alpine:latest as builder
RUN apk add --no-cache gcc g++ ninja cmake
WORKDIR /app
COPY . .
RUN cmake -Bbuild -S. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -mtune=skylake -march=core2" \
    && ninja -Cbuild

FROM alpine:latest as runtime
RUN apk add --no-cache libstdc++ libgomp libgcc
COPY --from=builder app/build/GTA_SA_cheat_finder .
# Test exec
RUN ./GTA_SA_cheat_finder --min 0 --max 396
ENTRYPOINT ["./GTA_SA_cheat_finder"] 
