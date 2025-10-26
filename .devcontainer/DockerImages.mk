#//////////////////////////////////////////////////////////////
#//                                                          //
#//  docker-multimedia, 2024                                 //
#//  Created: 30, May, 2021                                  //
#//  Modified: 14 November, 2024                             //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

SUBDIRS ?= debian ubuntu fedora archlinux

AUTHOR ?= bensuperpc
WEB_SITE ?= bensuperpc.org

# Base image
BASE_IMAGE_REGISTRY ?= docker.io

# Output docker image
OUTPUT_IMAGE_REGISTRY ?= docker.io
OUTPUT_IMAGE_PATH ?= bensuperpc
OUTPUT_IMAGE_NAME ?= multimedia
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

# Security
CAP_DROP ?= # --cap-drop ALL
CAP_ADD ?= # --cap-add SYS_PTRACE

comma?= ,
PLATFORMS ?= $(subst $() $(),$(comma),$(ARCH_LIST))

# Custom targets
CUSTOM_TARGET ?= help

# Git config
GIT_SHA ?= $(shell git rev-parse HEAD)
GIT_ORIGIN ?= $(shell git config --get remote.origin.url) 

DATE ?= $(shell date -u +"%Y%m%d")
UUID ?= $(shell uuidgen)

CURRENT_USER ?= $(shell whoami)
UID ?= $(shell id -u ${CURRENT_USER})
GID ?= $(shell id -g ${CURRENT_USER})
USERNAME ?= user

# Merge all variables
MAKEFILE_VARS ?= AUTHOR=$(AUTHOR) PLATFORMS="$(PLATFORMS)" \
	CPUS=$(CPUS) CPU_SHARES=$(CPU_SHARES) MEMORY=$(MEMORY) MEMORY_RESERVATION=$(MEMORY_RESERVATION) \
	BUILD_CPU_SHARES=$(BUILD_CPU_SHARES) BUILD_MEMORY=$(BUILD_MEMORY) WEB_SITE=$(WEB_SITE) \
	DOCKERFILE=$(DOCKERFILE) DOCKER_EXEC=$(DOCKER_EXEC) DOCKER_DRIVER=$(DOCKER_DRIVER) \
	GIT_SHA=$(GIT_SHA) GIT_ORIGIN=$(GIT_ORIGIN) DATE=$(DATE) UUID=$(UUID) \
	USERNAME=$(USERNAME) UID=$(UID) GID=$(GID) TMPFS_SIZE=$(TMPFS_SIZE) \
	TEST_CMD=$(TEST_CMD) RUN_CMD=$(RUN_CMD) PROGRESS_OUTPUT=$(PROGRESS_OUTPUT) \
	CAP_DROP=$(CAP_DROP) CAP_ADD=$(CAP_ADD) BASE_IMAGE_REGISTRY=$(BASE_IMAGE_REGISTRY) \
	OUTPUT_IMAGE_REGISTRY=$(OUTPUT_IMAGE_REGISTRY) OUTPUT_IMAGE_PATH=$(OUTPUT_IMAGE_PATH) \
	OUTPUT_IMAGE_NAME=$(OUTPUT_IMAGE_NAME) OUTPUT_IMAGE_VERSION=$(OUTPUT_IMAGE_VERSION)

default: all

.PHONY: $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) $(MAKEFILE_VARS) -C $@ all

.PHONY: all
all: $(addsuffix -all, $(SUBDIRS))

%.all:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.all,%,$@) all

.PHONY: build
build: $(addsuffix .build, $(SUBDIRS))

%.build:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.build,%,$@) build

.PHONY: test
test: $(addsuffix .test, $(SUBDIRS))

%.test:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.test,%,$@) test

.PHONY: run
run: $(addsuffix .run, $(SUBDIRS))

%.run:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.run,%,$@) run

.PHONY: version
version: $(addsuffix .version, $(SUBDIRS))

%.version:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.version,%,$@) version

.PHONY: clean
clean: $(addsuffix .clean, $(SUBDIRS))

%.clean:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.clean,%,$@) clean

.PHONY: purge
purge: $(addsuffix .purge, $(SUBDIRS))

%.purge:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.purge,%,$@) purge

.PHONY: update
update: $(addsuffix .update, $(SUBDIRS))

%.update:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.update,%,$@) update

.PHONY: push
push: $(addsuffix .push, $(SUBDIRS))

%.push:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.push,%,$@) push

.PHONY: pull
pull: $(addsuffix .pull, $(SUBDIRS))

%.pull:
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.pull,%,$@) pull

.PHONY: $(CUSTOM_TARGET)
$(CUSTOM_TARGET): $(addsuffix .$(CUSTOM_TARGET), $(SUBDIRS))

%.$(CUSTOM_TARGET):
	$(MAKE) $(MAKEFILE_VARS) -C $(patsubst %.$(CUSTOM_TARGET),%,$@) $(CUSTOM_TARGET)