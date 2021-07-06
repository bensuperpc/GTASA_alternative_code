#!/bin/bash
set -euo pipefail

cmake_build () {
    local image=$1
    shift 1
    local cmake_arg=$@
    echo "cmake arg: $cmake_arg"

    echo "Pulling dockcross/$image:latest"
    docker pull dockcross/$image:latest
    echo "Make script dockcross-$image"
    docker run --rm dockcross/$image:latest > ./dockcross-$image
    chmod +x ./dockcross-$image
    echo "Build build-$image"
    ./dockcross-$image cmake -Bbuild-$image -H. -GNinja $cmake_arg
    ./dockcross-$image ninja -Cbuild-$image
}
