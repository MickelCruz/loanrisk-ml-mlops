import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import (
    LoanRequest,
    LoanResponse,
    LoanExplainResponse,
    BatchRequest,
    BatchResponse
)
from api.predictor import predecir, predecir_batch, explicar

# ── Cargar configuración ───────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models'

with open(MODELS_DIR / 'resultados_finales.json') as f:
    config = json.load(f)

# ── Inicializar FastAPI ────────────────────────────────────────────────────────
app = FastAPI(
    title="LoanRisk-ML API",
    description="""
    API de scoring crediticio para préstamos personales.
    
    Predice la probabilidad de default y genera un score 300-850
    similar al score FICO usando un modelo XGBoost entrenado
    con datos de Lending Club.
    """,
    version="1.0.0"
)


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """
    Verifica que la API está funcionando correctamente.
    """
    return {
        "status":  "ok",
        "modelo":  config['modelo'],
        "version": "1.0.0"
    }


# ── Endpoint de scoring individual ────────────────────────────────────────────
@app.post("/score", response_model=LoanResponse)
def score(request: LoanRequest):
    """
    Genera el score crediticio para un préstamo individual.

    - **score**: score 300-850 (mayor = menor riesgo)
    - **probabilidad_default**: probabilidad de que el préstamo defaultee
    - **decision**: no_default o default
    - **riesgo**: bajo, medio o alto
    """
    try:
        data      = request.model_dump()
        resultado = predecir(data)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint de scoring batch ─────────────────────────────────────────────────
@app.post("/score/batch", response_model=BatchResponse)
def score_batch(request: BatchRequest):
    """
    Genera scores crediticios para múltiples préstamos.
    Útil para scoring de portfolios completos.
    """
    try:
        registros  = [p.model_dump() for p in request.prestamos]
        resultados = predecir_batch(registros)
        return {
            "resultados": resultados,
            "total":      len(resultados)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint de explicabilidad ────────────────────────────────────────────────
@app.post("/explain", response_model=LoanExplainResponse)
def explain(request: LoanRequest):
    """
    Genera el score crediticio y explica las top 5 features
    que más influyeron en la predicción usando SHAP.

    - **top_features**: lista de features con su contribución y dirección
    - **direccion**: aumenta_riesgo o reduce_riesgo
    """
    try:
        data      = request.model_dump()
        resultado = explicar(data)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint de información del modelo ────────────────────────────────────────
@app.get("/model/info")
def model_info():
    """
    Retorna información del modelo en producción —
    métricas, threshold y configuración del scorecard.
    """
    return {
        "modelo":        config['modelo'],
        "threshold":     config['threshold'],
        "metricas":      config['metricas'],
        "scorecard":     config['scorecard'],
        "escala_riesgo": config['escala_riesgo']
    }