# Stage 1 - build
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev cmake build-essential \
    libgraphicsmagick1-dev libatlas-base-dev \
    libhdf5-dev pkg-config git gcc-12 g++-12 \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel scikit-build

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    Pillow \
    opencv-python-headless \
    requests \
    tqdm \
    gdown \
    Flask \
    flask_cors \
    gunicorn \
    retina-face \
    insightface \
    tensorflow==2.19.0 \
    onnxruntime-gpu \
    tf-keras \
    typing-extensions \
    pydantic

# Clone fork and install the deepface package itself
RUN git clone https://github.com/enoquelights/deepface.git /app
RUN pip install --no-cache-dir -e /app

# Stage 2 - runtime only
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git \
    libsm6 libxext6 libxrender1 \
    libgraphicsmagick1-dev \
    libhdf5-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the app from builder (already cloned and installed)
COPY --from=builder /app /app

RUN chown -R 1001:0 /app

#WORKDIR /app/deepface/api/src
WORKDIR /app/api/src
EXPOSE 5000
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["sh", "/app/entrypoint.sh"]