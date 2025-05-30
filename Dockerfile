# Use a specific version and distribution of Python runtime as a parent image
FROM python:3.10.13-bookworm

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Define Calibre version and installation directory
ARG CALIBRE_VERSION=8.4.0
ENV CALIBRE_INSTALL_DIR="/opt/calibre"

# Install necessary system dependencies, including openssl and CA certificates
# Group system updates and installations together for better layer management.
# This layer will be cached if the list of packages or their versions don't change.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    unzip \
    openssl \
    # For Calibre installer and runtime
    python3-lxml \
    python3-mechanize \
    python3-pyqt5.qtwebengine \
    python3-pyqt5 \
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
    libtiff6 \
    libwebp7 \
    poppler-utils \
    libxml2-dev \
    libxslt1-dev \
    libgtk-3-0 \
    libglib2.0-0 \
    libglib2.0-data \
    libice6 \
    libtiff-dev \
    libpng-dev \
    # For running the installer script and NLTK downloads
    ca-certificates \
    xz-utils \
    # Update CA certificates after installation
    && update-ca-certificates --fresh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set SSL environment variables after ca-certificates are processed
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs

# Install specific version of Calibre using the official installer
# This layer will be cached if CALIBRE_VERSION doesn't change.
RUN wget -nv -O calibre-installer.py https://download.calibre-ebook.com/linux-installer.py && \
    python3 calibre-installer.py \
        --version ${CALIBRE_VERSION} \
        --isolated \
        --installation-dir ${CALIBRE_INSTALL_DIR} && \
    rm calibre-installer.py

# Ensure that ebook-convert and Calibre libraries are findable
ENV PATH="${CALIBRE_INSTALL_DIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CALIBRE_INSTALL_DIR}/lib"

# Verify Calibre installation (optional, but good for debugging early)
RUN echo "Verifying Calibre installation..." && ebook-convert --version

# Set working directory for the application
WORKDIR /app

# Create necessary application directories. This is unlikely to change often.
RUN mkdir -p /app/Working_files/Book /app/Working_files/temp_ebook /app/Working_files/temp

# Copy only the requirements file first.
# This is crucial for caching the pip install layer.
COPY requirements.txt ./

# Install Python dependencies.
# This layer will ONLY be rebuilt if requirements.txt changes, or any layer above it changes.
# Using --no-cache-dir reduces layer size but means pip doesn't use its own HTTP cache for this run.
# For retaining pip's own download cache across different builds (if requirements.txt changes slightly),
# consider Docker BuildKit's cache mounts: RUN --mount=type=cache,target=/root/.cache/pip pip install ...
RUN pip install --upgrade pip==24.1.2
RUN pip install --verbose --no-cache-dir -r requirements.txt

# Set NLTK data directory (depends on nltk being installed from requirements.txt)
ENV NLTK_DATA="/usr/share/nltk_data"
RUN mkdir -p "${NLTK_DATA}" && echo "NLTK_DATA directory ${NLTK_DATA} created."

# Download NLTK punkt data. This runs after NLTK is installed.
RUN echo "Attempting to download NLTK punkt using CLI..." && \
    python -m nltk.downloader -d "${NLTK_DATA}" punkt && \
    echo "NLTK punkt download command executed."

# Verify NLTK punkt installation
RUN echo "Verifying NLTK punkt installation in ${NLTK_DATA}/tokenizers/punkt..." && \
    if [ -d "${NLTK_DATA}/tokenizers/punkt" ]; then \
        echo "NLTK punkt is correctly installed in ${NLTK_DATA}/tokenizers/punkt." && \
        ls -lA "${NLTK_DATA}/tokenizers/punkt/"; \
    elif [ -f "${NLTK_DATA}/tokenizers/punkt.zip" ]; then \
        echo "Found ${NLTK_DATA}/tokenizers/punkt.zip. Unzipping..." && \
        unzip "${NLTK_DATA}/tokenizers/punkt.zip" -d "${NLTK_DATA}/tokenizers/" && \
        if [ -d "${NLTK_DATA}/tokenizers/punkt" ]; then \
            echo "Unzipped successfully. NLTK punkt is now in ${NLTK_DATA}/tokenizers/punkt." && \
            ls -lA "${NLTK_DATA}/tokenizers/punkt/"; \
        else \
            echo "Error: Failed to find ${NLTK_DATA}/tokenizers/punkt after unzipping." >&2; \
            exit 1; \
        fi; \
    else \
        echo "Error: NLTK punkt data (directory or zip) not found in ${NLTK_DATA}/tokenizers." >&2; \
        exit 1; \
    fi

# ---- APPLICATION CODE AND ASSETS ----
# Copy your application code and assets AFTER installing dependencies.
# Changes to these files will now only invalidate these COPY layers and subsequent RUN layers.

# Copy the extracted working f5-tts code.
# Consider if this truly needs to be in site-packages or if it can live in /app
# and be imported from there. If it's a local package with setup.py, `pip install ./f5_tts_working_code`
# would be more standard after copying it in.
COPY ./f5_tts_working_code /usr/local/lib/python3.10/site-packages/f5_tts

COPY default_voice.mp3 /app/default_voice.mp3
COPY preload_models.py /app/preload_models.py

# Run preload_models.py AFTER it's copied and dependencies are installed.
RUN python /app/preload_models.py

# Copy the main application file last, as it might change most frequently.
COPY app.py .

# Set ENTRYPOINT and CMD
ENTRYPOINT ["python", "app.py"]
CMD []