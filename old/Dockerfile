FROM alpine:latest as builder
RUN apk add --no-cache gcc g++ ninja cmake
WORKDIR /app
COPY . .
RUN cmake -Bbuild -H. -GNinja -DCMAKE_BUILD_TYPE=Release && ninja -Cbuild

FROM alpine:latest as runtime
RUN apk add --no-cache libstdc++ libgomp libgcc
COPY --from=builder app/build/gta .
# Test exec
RUN ./gta 0 52
ENTRYPOINT ["./gta"] 
