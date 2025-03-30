# Use a slim version of Python runtime as a parent image
FROM python:3.10.13

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install necessary system dependencies - Part 1: Update and Install
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    libegl1 \
    libgl1 \
    # libgl1-mesa-glx
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
    # libxrender1
    # libxfixes3
    # libxdamage1
    # libxext6
    # libsm6
    # libx11-6
    # libxft2
    # libxinerama1
    # libxrandr2
    # libxcomposite1
    # libxcursor1
    # libxi6
    # libfontconfig1
    # libfreetype6
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
    libxslt1-dev      # NO backslash here, end of install list
    # libgtk-3-0
    # libglib2.0-0
    # libglib2.0-data
    # libice6

# Install necessary system dependencies - Part 2: Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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

COPY default_voice.mp3 /app/default_voice.mp3

# Install Python dependencies (BEFORE downloading NLTK data)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --verbose -r requirements.txt

# --- Robust NLTK Data Download ---
# Explicitly set the download directory and download 'punkt'
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt

# --->>> ADDED: List contents of downloaded punkt directory <<<---
RUN echo "Listing NLTK punkt data directory contents:" && \
    ls -lR /usr/local/share/nltk_data/tokenizers/punkt || echo "Punkt directory not found or ls failed"

# Set the NLTK_DATA environment variable so NLTK knows where to look
ENV NLTK_DATA=/usr/local/share/nltk_data:/usr/share/nltk_data:/root/nltk_data
# --- End NLTK Data Download ---

# Copy preload_models.py AFTER NLTK data is needed/downloaded
COPY preload_models.py /app/preload_models.py
RUN python /app/preload_models.py

# Copy your application files
COPY app.py .

# **Set ENTRYPOINT and CMD**
ENTRYPOINT ["python", "app.py"]
CMD []