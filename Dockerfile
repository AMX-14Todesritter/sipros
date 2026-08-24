FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    openmpi-bin \
    libopenmpi-dev \
    gdb \
    google-perftools \
    libgoogle-perftools-dev \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    pip3 install lxml pandas

WORKDIR /workspace/sipros

CMD ["/bin/bash"]