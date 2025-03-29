# Use a slim version of Python runtime as a parent image
FROM python:3.10.13

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    libegl1 \
    libgl1 \
    libgl1-mesa-glx \
    libopengl0 \
    libxcb-cursor0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-glx0 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libx11-xcb1 \
    libxrender1 \
    libxfixes3 \
    libxdamage1 \
    libxext6 \
    libsm6 \
    libx11-6 \
    libxft2 \
    libxinerama1 \
    libxrandr2 \
    libxcomposite1 \
    libxcursor1 \
    libxi6 \
    libfontconfig1 \
    libfreetype6 \
    libssl3 \
    libxml2 \
    libxslt1.1 \
    libsqlite3-0 \
    zlib1g \
    libopenjp2-7 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff-dev \
    libwebp7 \
    poppler-utils \
    libxml2-dev \
    libxslt1-dev \
    libgtk-3-0 \
    libglib2.0-0 \
    libglib2.0-data \
    libice6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Calibre (which includes ebook-convert)
RUN wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | sh /dev/stdin

# Ensure that ebook-convert is available in PATH
ENV PATH="/root/calibre:${PATH}"

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/Working_files/Book /app/Working_files/temp_ebook /app/Working_files/temp

# Copy the requirements file
COPY requirements.txt .

# Copy the extracted working f5-tts code into the image
COPY ./f5_tts_working_code /usr/local/lib/python3.10/site-packages/f5_tts

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt

# Copy your application files
COPY app.py .

# **Set ENTRYPOINT and CMD**
ENTRYPOINT ["python", "app.py"]
CMD []
