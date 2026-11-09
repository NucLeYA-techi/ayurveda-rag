FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs
COPY benchmarks ./benchmarks
EXPOSE 8080
CMD ["ayurrag", "serve", "--host", "0.0.0.0", "--port", "8080"]
