FROM python:3.12-slim

# Dependências de sistema (apenas curl para healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependências primeiro (camada cacheável)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY app/ ./app/

EXPOSE 8000

# 2 workers: adequado para 1 vCPU da VPS Hostinger.
# Ajustar para 4 se escalar para 2 vCPUs.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--access-log"]
