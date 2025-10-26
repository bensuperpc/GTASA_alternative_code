FROM alpine:latest as builder
RUN apk add --no-cache gcc g++ ninja cmake
WORKDIR /app
COPY . .
RUN cmake -Bbuild -H. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -mtune=skylake -march=core2" \
    && ninja -Cbuild

FROM alpine:latest as runtime
RUN apk add --no-cache libstdc++ libgomp libgcc
COPY --from=builder app/build/GTASA_alternative_code .
# Test exec
RUN ./GTASA_alternative_code --min 0 --max 396
ENTRYPOINT ["./GTASA_alternative_code"] 
