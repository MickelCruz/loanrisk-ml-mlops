#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# entrypoint.sh — LoanRisk-ML API
# ══════════════════════════════════════════════════════════════════════════════

set -e

MODELS_DIR="/app/models"
REQUIRED_FILES=(
    "XGBoost_best.joblib"
    "preprocessor.joblib"
    "shap_explainer.joblib"
    "resultados_finales.json"
    "valores_validos.json"
    "best_params.json"
)

# ── Descargar modelos desde GCS si aplica ─────────────────────────────────────
if [ -n "$GCS_BUCKET" ]; then
    echo "GCS_BUCKET detectado: $GCS_BUCKET"
    echo "Descargando modelos desde GCS..."

    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$MODELS_DIR/$file" ]; then
            echo "  Descargando $file..."
            python -c "
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('$GCS_BUCKET')
blob = bucket.blob('models/$file')
blob.download_to_filename('$MODELS_DIR/$file')
print('  OK: $file')
"
        else
            echo "  OK (ya existe): $file"
        fi
    done
    echo "Modelos listos."

else
    echo "GCS_BUCKET no definido — usando modelos locales."

    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$MODELS_DIR/$file" ]; then
            echo "ERROR: Modelo no encontrado: $MODELS_DIR/$file"
            echo "Monta el directorio models/ como volumen o define GCS_BUCKET."
            exit 1
        fi
    done
    echo "Modelos locales verificados."
fi

# ── Arrancar la API ───────────────────────────────────────────────────────────
echo "Iniciando LoanRisk-ML API en puerto ${PORT:-8080}..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --workers 1