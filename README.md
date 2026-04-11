# LoanRisk-ML — Sistema de Scoring Crediticio

![CI](https://github.com/MickelCruz/loanrisk-ml-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/api-FastAPI-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)

Sistema end-to-end de scoring crediticio para préstamos personales, entrenado con el dataset de Lending Club (2.26M registros). Predice la probabilidad de default y genera un score 300-850 similar al score FICO usando XGBoost con un pipeline MLOps completo.

---

## Stack Técnico

| Categoría | Tecnologías |
|---|---|
| Modelo | XGBoost, LightGBM, Optuna, SHAP |
| API | FastAPI, Pydantic v2, Uvicorn |
| Pipelines | Prefect |
| Monitoring | Evidently |
| Infraestructura | Docker, GitHub Actions |
| Experiment Tracking | MLflow |

---

## Métricas del Modelo

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.787 |
| AUC-PR | 0.570 |
| KS | 0.417 |
| Gini | 0.573 |
| Threshold | 0.5387 |

---

## Quickstart

```bash
# Clonar
git clone https://github.com/MickelCruz/loanrisk-ml-mlops.git
cd loanrisk-ml-mlops

# Correr con Docker
docker-compose up --build

# Correr tests
pytest tests/ -v
```

API disponible en `http://localhost:8080/docs`

---

## Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Estado de la API |
| GET | `/model/info` | Métricas y configuración |
| POST | `/score` | Score individual |
| POST | `/score/batch` | Score de múltiples préstamos |
| POST | `/explain` | Score + explicación SHAP |

**Ejemplo de respuesta `/score`:**
```json
{
  "probabilidad_default": 0.2134,
  "score": 733,
  "decision": "no_default",
  "riesgo": "bajo"
}
```

---

## CI/CD

Cada push a `main` dispara automáticamente los 33 tests via GitHub Actions.
Los modelos reales viven fuera del repo — el CI usa modelos mock con la misma estructura.

---

## Decisiones Técnicas

- **XGBoost sobre LightGBM** — mismo AUC-PR pero menor gap train-val (0.059 vs 0.081)
- **AUC-PR como métrica principal** — más informativa que AUC-ROC para datasets desbalanceados (~20% default)
- **Threshold 0.5387** — optimizado por F1-score sobre el conjunto de test
- **Champion/challenger** — el pipeline de reentrenamiento solo reemplaza el modelo si mejora el AUC-PR Val

---

## Autor

**Mickel Cruz** — ML/MLOps Engineer · [GitHub](https://github.com/MickelCruz)