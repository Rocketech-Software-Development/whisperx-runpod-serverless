# Official WhisperX v3.4.2 for RunPod Serverless
# Using the latest official source from https://github.com/m-bain/whisperX
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables for CUDA and container
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Model and cache directories
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

# Install system dependencies (using Python 3.10 for Ubuntu 22.04)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlinks for Python
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Upgrade pip and install uv for faster package management
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir uv

# Install PyTorch with CUDA 12.1 support (install first for better caching)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchaudio==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install working tokenizers version first to avoid puccinialin issue
RUN pip install --no-cache-dir tokenizers==0.15.2

# Install faster-whisper with fixed tokenizers
RUN pip install --no-cache-dir --no-deps faster-whisper==1.0.3

# Install missing dependencies for faster-whisper manually
RUN pip install --no-cache-dir av huggingface-hub ctranslate2

# Install only essential missing dependencies (lighter image)
RUN pip install --no-cache-dir transformers>=4.30.0 librosa soundfile pyannote.audio

# Install WhisperX from latest and fix compatibility in handler
RUN pip install --no-cache-dir --no-deps --no-build-isolation git+https://github.com/m-bain/whisperX.git

# Install RunPod SDK and additional dependencies
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    requests>=2.31.0 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    aiofiles>=23.2.1 \
    pydantic>=2.5.0

# Copy application files
COPY handler.py /app/
COPY requirements.txt /app/

# Create cache directories
RUN mkdir -p /app/cache /app/models /tmp/whisperx

# Create model download script
RUN echo '#!/usr/bin/env python3\n\
import whisperx\n\
import torch\n\
device = "cuda" if torch.cuda.is_available() else "cpu"\n\
print(f"Pre-downloading models on {device}...")\n\
try:\n\
    model = whisperx.load_model("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")\n\
    print("✅ Whisper large-v3 model downloaded")\n\
    del model\n\
    if device == "cuda": torch.cuda.empty_cache()\n\
    for lang in ["en", "ru", "de", "fr", "es"]:\n\
        try:\n\
            align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)\n\
            print(f"✅ {lang} alignment model downloaded")\n\
            del align_model, metadata\n\
            if device == "cuda": torch.cuda.empty_cache()\n\
        except Exception as e:\n\
            print(f"⚠️ Could not download {lang} alignment model: {e}")\n\
    print("🎉 Model pre-download completed!")\n\
except Exception as e:\n\
    print(f"⚠️ Model pre-download failed: {e}")\n\
    print("Models will be downloaded on first use")\n\
' > /app/download_models.py && chmod +x /app/download_models.py

# Skip model pre-download to keep image lighter - models will download on first use
# RUN python3 /app/download_models.py || echo "Model pre-download failed, will download on first use"

# Health check to verify CUDA and WhisperX
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; import whisperx; print('🔍 Health Check:'); print(f'CUDA available: {torch.cuda.is_available()}'); print('WhisperX imported successfully')" || exit 1

# Command to run the RunPod handler
CMD ["python3", "-u", "handler.py"] 