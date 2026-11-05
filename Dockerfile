FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY drishti ./drishti
COPY models ./models
RUN pip install --no-cache-dir ".[fast]"
ENV DRISHTI_DB=/data/drishti.db DRISHTI_MODELS=/app/models
VOLUME /data
EXPOSE 8000
CMD ["drishti", "serve", "--host", "0.0.0.0", "--port", "8000"]
