FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
ENV RP_HOST=0.0.0.0 RP_UI_PORT=8847 RP_PORT=8627 RP_DATA_DIR=/data HF_HOME=/data/huggingface
ENV OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && python -m pip install -r requirements.txt
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin roleplay && mkdir -p /data && chown roleplay:roleplay /data /app
COPY --chown=roleplay:roleplay . .
USER roleplay
EXPOSE 8847
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8847/", timeout=3).read(1)'
CMD ["/bin/sh", "scripts/launch-rp.sh"]
