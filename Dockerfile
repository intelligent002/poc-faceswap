FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# ===== Install system deps =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-imgproc-dev \
    libopencv-videoio-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ===== Create working directory =====
WORKDIR /app

# ===== Copy Python dependencies =====
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ===== Copy application =====
COPY models ./models
COPY app ./app

# ===== Expose port =====
EXPOSE 8000

# ===== Runtime arguments (optimize threading) =====
ENV OMP_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

# ===== Run server =====
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
