FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /usr/src/app

# -------------------------
# Environment variables
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/usr/src/app:$PYTHONPATH
ENV HF_HOME=/usr/src/app/models
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# -------------------------
# Install system dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    ffmpeg \
    gcc \
    curl \
    libldap2-dev \
    libsasl2-dev \
    libssl-dev \
    git \
    openssl \
    ca-certificates \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavdevice-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Upgrade pip and install Python dependencies
# -------------------------
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install torch==2.5.1 torchvision==0.20.1
RUN pip install --no-cache-dir --index-url https://pypi.org/simple/ -r requirements.txt

# -------------------------
# Create static and media directories
# -------------------------
RUN mkdir -p /usr/src/app/staticfiles /usr/src/app/mediafiles /tmp/django_uploads && chmod 1777 /tmp/django_uploads

# -------------------------
# Expose port
# -------------------------
EXPOSE 8000

# -------------------------
# Copy entrypoint script
# -------------------------
COPY ./entrypoint.sh .
RUN sed -i 's/\r$//g' /usr/src/app/entrypoint.sh
RUN chmod +x /usr/src/app/entrypoint.sh

# -------------------------
# Set entrypoint
# -------------------------
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
