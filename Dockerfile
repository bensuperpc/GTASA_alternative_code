FROM alpine:latest as builder
RUN apk add --no-cache gcc g++ ninja cmake
WORKDIR /app
COPY . .
RUN cmake -Bbuild -H. -GNinja -DCMAKE_BUILD_TYPE=Release && ninja -Cbuild

FROM alpine:latest as runtime
RUN apk add --no-cache libstdc++ libgomp libgcc
COPY --from=builder app/build/GTA_SA_cheat_finder .
# Test exec
RUN ./GTA_SA_cheat_finder 0 52
ENTRYPOINT ["./GTA_SA_cheat_finder"] 
