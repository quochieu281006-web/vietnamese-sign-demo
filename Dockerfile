FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

COPY src/ ./src/
COPY web/ ./web/
COPY models/VSL_unified_static.joblib ./models/VSL_unified_static.joblib
COPY models/VSL_tone_rf.joblib ./models/VSL_tone_rf.joblib

EXPOSE 10000
CMD ["sh", "-c", "exec python src/web_app.py --host 0.0.0.0 --port ${PORT:-10000}"]
