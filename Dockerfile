FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    openmpi-bin \
    libopenmpi-dev \
    gdb \
    libgoogle-perftools-dev \
    python3 \
    python3-pip \
    python3-venv \
    seqkit \
    git \
    wget \
    curl \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    lxml==4.9.4 \
    pandas==2.2.3

WORKDIR /workspace/sipros

CMD ["/bin/bash"]
