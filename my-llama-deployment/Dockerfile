# Dockerfile for GPU Inference with PyTorch 2.7.0 and CUDA 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set env for PyTorch and HuggingFace
ENV DEBIAN_FRONTEND=noninteractive \
    TRANSFORMERS_CACHE=/opt/ml/model/cache \
    HF_HOME=/opt/ml/model/cache \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl python3-pip python3-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Use Python 3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and set working dir
RUN pip install --upgrade pip
WORKDIR /opt/ml/model

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only inference logic, NOT the model itself
COPY inference.py .

# SageMaker expects this setup
ENV SAGEMAKER_PROGRAM inference.py
ENTRYPOINT ["python", "inference.py"]
CMD ["serve"]
