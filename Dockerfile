FROM python:3.11.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml requirements.lock README.md ./
COPY config ./config
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock \
    && pip install --no-cache-dir --no-deps .

CMD ["uvicorn", "btc_signal.api:app", "--host", "0.0.0.0", "--port", "8000"]
