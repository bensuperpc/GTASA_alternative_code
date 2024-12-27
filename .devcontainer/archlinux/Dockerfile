# Base image
ARG DOCKER_IMAGE=archlinux:base
FROM ${DOCKER_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN pacman -Syu --noconfirm && pacman -S --noconfirm \
    # Essential packages
    base-devel git cmake extra-cmake-modules ninja automake autoconf \
    libtool pkgconf \ 
    openssl ccache bash \
    && pacman -Scc --noconfirm

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen en_US.UTF-8 \
    localectl set-locale LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Buider image
FROM base AS builder

RUN pacman -Syu --noconfirm && pacman -S --noconfirm \
# All needed packages
    util-linux mesa meson clang llvm ca-certificates \
    && pacman -Scc --noconfirm

WORKDIR /work

# Final image
FROM base AS final

ARG BUILD_DATE=""
ARG VCS_REF=""
ARG VCS_URL=""
ARG PROJECT_NAME=""
ARG AUTHOR=""
ARG URL=""

ARG IMAGE_VERSION="1.0.0"
ENV IMAGE_VERSION=${IMAGE_VERSION}

ARG CCACHE_MAXSIZE=16G
ENV CCACHE_MAXSIZE=${CCACHE_MAXSIZE}

ENV TERM=xterm-256color

LABEL maintainer="Bensuperpc"
LABEL author="Bensuperpc"
LABEL description="Docker image for benlib"

LABEL org.label-schema.schema-version="1.0" \
      org.label-schema.build-date=${BUILD_DATE} \
      org.label-schema.name=${PROJECT_NAME} \
      org.label-schema.description="benlib" \
      org.label-schema.version=${IMAGE_VERSION} \
      org.label-schema.vendor=${AUTHOR} \
      org.label-schema.url=${URL} \
      org.label-schema.vcs-url=${VCS_URL} \
      org.label-schema.vcs-ref=${VCS_REF} \
      org.label-schema.docker.cmd=""

VOLUME [ "/work" ]
WORKDIR /work

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ARG USERNAME=user
RUN useradd -m "${USERNAME}"
# Add color to bash
RUN echo PS1="'\[\033[01;32m\][\u@\h\[\033[01;37m\] \W\[\033[01;32m\]]\$\[\033[00m\] '" >> "/home/${USERNAME}/.bashrc"

ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/bin/bash" , "-l" ]
