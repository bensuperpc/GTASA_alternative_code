#//////////////////////////////////////////////////////////////
#//                                                          //
#//  docker-benlib, 2023                                     //
#//  Created: 04 February, 2023                              //
#//  Modified: 16 June, 2024                                 //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

# Base image
BASE_IMAGE_REGISTRY ?= docker.io
BASE_IMAGE_PATH ?= 
BASE_IMAGE_NAME ?= archlinux
BASE_IMAGE_TAGS ?= base

# Output docker image
AUTHOR ?= bensuperpc
WEB_SITE ?= bensuperpc.org

OUTPUT_IMAGE_REGISTRY ?= docker.io
OUTPUT_IMAGE_PATH ?= bensuperpc
OUTPUT_IMAGE_NAME ?= gtasa_alternative_code
OUTPUT_IMAGE_VERSION ?= 1.0.0

# Max CPU and memory
CPUS ?= 8.0
CPU_SHARES ?= 1024
MEMORY ?= 16GB
MEMORY_RESERVATION ?= 2GB
TMPFS_SIZE ?= 4GB
BUILD_CPU_SHARES ?= 1024
BUILD_MEMORY ?= 16GB

TEST_CMD ?= ./test/test.sh
RUN_CMD ?=

include DockerImage.mk