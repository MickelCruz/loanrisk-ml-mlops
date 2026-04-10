# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — LoanRisk-ML API
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements-api.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copiar código y modelos
COPY api/       ./api/
COPY models/    ./models/
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

# Usuario no-root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]