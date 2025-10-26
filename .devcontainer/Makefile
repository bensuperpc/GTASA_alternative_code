#//////////////////////////////////////////////////////////////
#//                                                          //
#//  docker-benlib, 2023                                         //
#//  Created: 04 February, 2023                              //
#//  Modified: 05 October, 2024                              //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

SUBDIRS ?= archlinux

AUTHOR ?= bensuperpc
WEB_SITE ?= bensuperpc.org

# Base image
BASE_IMAGE_REGISTRY ?= docker.io

# Output docker image
OUTPUT_IMAGE_REGISTRY ?= docker.io
OUTPUT_IMAGE_PATH ?= bensuperpc
OUTPUT_IMAGE_NAME ?= gtasa_alternative_code
OUTPUT_IMAGE_VERSION ?= 1.0.0

TEST_CMD ?= ./test/test.sh
RUN_CMD ?= 

# Docker config
DOCKERFILE ?= Dockerfile
DOCKER_EXEC ?= docker
PROGRESS_OUTPUT ?= plain

# --push
DOCKER_DRIVER ?= --load
ARCH_LIST ?= linux/amd64

# Max CPU and memory
CPUS ?= 8.0
CPU_SHARES ?= 1024
MEMORY ?= 16GB
MEMORY_RESERVATION ?= 2GB
TMPFS_SIZE ?= 4GB
BUILD_CPU_SHARES ?= 1024
BUILD_MEMORY ?= 16GB


include DockerImages.mk
