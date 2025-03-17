FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
ENV PYTHONBUFFERING=1
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi