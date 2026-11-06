# FlowCast: dashboard and pipeline in one image.
#   docker build -t flowcast .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" -v "$PWD/reports:/app/reports" \
#       flowcast python -m src.pipeline --data-dir data/raw          # train (about 30 min)
#   docker run -d -p 8501:8501 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
#       -v "$PWD/reports:/app/reports" --name flowcast flowcast      # serve the dashboard
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime used by XGBoost. curl: container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .
RUN useradd --create-home --uid 10001 flowcast \
    && mkdir -p data/processed data/interim models reports/figures \
    && chown -R flowcast:flowcast /app
USER flowcast

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
