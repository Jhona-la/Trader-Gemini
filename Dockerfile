# ==========================================
# STAGE 1: BUILDER (Compilers & Heavy Libs)
# ==========================================
FROM python:3.11-slim-bookworm as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1. Install System Dependencies (GCC for TA-Lib, Numba, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# 2. Compile TA-Lib (C Library)
# TA-Lib is required by the python wrapper
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 3. Install Python Dependencies
# We install into a virtual environment to easily copy it later
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 2: RUNTIME (Secure & Slim)
# ==========================================
FROM python:3.11-slim-bookworm as runtime

# Security: Create non-root user
RUN groupadd -r gemini && useradd -r -g gemini gemini

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

WORKDIR /app

# 1. Copy TA-Lib Shared Libraries from Builder
COPY --from=builder /usr/lib/libta_lib.* /usr/lib/
COPY --from=builder /usr/include/ta-lib /usr/include/ta-lib

# 2. Copy Virtual Environment from Builder
COPY --from=builder /opt/venv /opt/venv

# 3. Copy Application Code
COPY . .

# 4. Permissions & Cleanup
# Ensure the non-root user owns the app directory
RUN chown -R gemini:gemini /app && \
    # Remove potentially sensitive files if they slipped via COPY
    rm -f .env

# 5. Switch to Non-Root User
USER gemini

# 6. Healthcheck (Basic process check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD ps aux | grep "[m]ain.py" || exit 1

# 7. Entrypoint
# We assume main.py is the entrypoint
CMD ["python", "main.py"]
