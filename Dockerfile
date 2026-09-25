# The training profile is one node with four accelerators and 80 GB of memory per
# node (Methods Sec. 4.7), so the image is pinned to a CUDA 12.4 runtime whose torch
# build matches the pinned version in requirements.txt. A CPU wheel path is
# documented in the README for hosts without an accelerator.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONHASHSEED=0 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y python3.11 python3-pip python3.11-venv git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/stagefm

COPY requirements.txt ./
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.11.0 \
    && python3.11 -m pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.txt ./
COPY configs ./configs
COPY scripts ./scripts
COPY src ./src
COPY tests ./tests

ENV PYTHONPATH=/opt/stagefm/src

CMD ["python3.11", "-m", "pytest", "-q", "tests"]
