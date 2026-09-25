# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Logs: one JSON object per line on stdout (LOG_LEVEL=debug|info|warning|error).
ENV LOG_LEVEL=info \
    LOG_FORMAT=json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV / MediaPipe runtime libraries. MediaPipe's native library links against
# EGL and GLES: without libegl1 / libgles2 the image builds and starts, and then
# every session fails on "libEGL.so.1: cannot open shared object file".
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch: the analysis service runs on CPU, and the default wheel
# drags in several GB of CUDA libraries it will never use.
COPY requirements-service.txt ./
# Generous timeouts/retries: these wheels are large and build networks are not always fast.
RUN pip install --default-timeout=120 --retries 8 \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-service.txt

COPY . .

# Fetch and hash-verify the identity / speaker models at build time so a running
# container never depends on a third-party download (model_assets.py checks
# each file against a pinned SHA-256).
RUN python -c "import model_assets as m; [m.ensure_asset(a) for a in (m.YUNET, m.SFACE, m.WESPEAKER)]; print('models ok')"

# Never run as root; models/ must stay readable.
RUN useradd --create-home --uid 10001 proctor && chown -R proctor /app
USER proctor

EXPOSE 8901
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8901/health', timeout=8).status==200 else 1)"

# NOTE: this service has no authentication. Keep it on the private network
# (reachable only by shepherd-backend), never on a public domain.
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8901"]
