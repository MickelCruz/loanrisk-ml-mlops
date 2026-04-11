import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

from prefect import task, flow, get_run_logger
from api.predictor import prob_a_score, clasificar_riesgo

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / 'data' / 'processed'
MODELS_DIR     = ROOT / 'models'
REPORTS_DIR    = ROOT / 'reports'


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="cargar_modelo_produccion")
def cargar_modelo_produccion():
    """Carga el modelo y preprocesador en producción."""
    logger = get_run_logger()

    model        = joblib.load(MODELS_DIR / 'XGBoost_best.joblib')
    preprocessor = joblib.load(MODELS_DIR / 'preprocessor.joblib')

    with open(MODELS_DIR / 'resultados_finales.json') as f:
        config = json.load(f)

    threshold = config['threshold']

    logger.info(f"Modelo cargado: {config['modelo']}")
    logger.info(f"Threshold: {threshold}")

    return model, preprocessor, threshold


@task(name="cargar_portfolio")
def cargar_portfolio() -> pd.DataFrame:
    """
    Carga el portfolio de préstamos a scorear.
    En producción real este sería un archivo nuevo que llega periódicamente.
    Para este pipeline usamos X_test como simulación del portfolio.
    """
    logger = get_run_logger()

    df = pd.read_parquet(DATA_PROCESSED / 'X_test.parquet')
    logger.info(f"Portfolio cargado: {df.shape[0]:,} préstamos")

    return df


@task(name="generar_scores")
def generar_scores(model, preprocessor, threshold: float,
                   df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera scores para todo el portfolio.
    Usa prob_a_score y clasificar_riesgo de predictor.py
    para garantizar consistencia con la API.
    """
    logger = get_run_logger()

    X      = preprocessor.transform(df)
    probas = model.predict_proba(X)[:, 1]

    scores = [prob_a_score(p) for p in probas]

    resultados = pd.DataFrame({
        'probabilidad_default': probas.round(4),
        'score':                scores,
        'decision':             ['default' if p >= threshold else 'no_default' for p in probas],
        'riesgo':               [clasificar_riesgo(s) for s in scores]
    })

    n_total    = len(resultados)
    n_default  = (resultados['decision'] == 'default').sum()
    n_aprobado = (resultados['decision'] == 'no_default').sum()

    logger.info(f"Scores generados: {n_total:,} préstamos")
    logger.info(f"  Aprobados:  {n_aprobado:,} ({n_aprobado/n_total:.1%})")
    logger.info(f"  Rechazados: {n_default:,} ({n_default/n_total:.1%})")
    logger.info(f"  Score promedio: {resultados['score'].mean():.0f}")

    return resultados


@task(name="guardar_resultados")
def guardar_resultados(resultados: pd.DataFrame):
    """Guarda los resultados del scoring batch."""
    logger = get_run_logger()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta      = REPORTS_DIR / f'batch_scores_{timestamp}.parquet'

    resultados.to_parquet(ruta, index=False)
    logger.info(f"Resultados guardados en: {ruta}")

    return ruta


@task(name="generar_reporte_batch")
def generar_reporte_batch(resultados: pd.DataFrame):
    """Genera un resumen ejecutivo del scoring batch."""
    logger = get_run_logger()

    n_total    = len(resultados)
    n_default  = (resultados['decision'] == 'default').sum()
    n_aprobado = (resultados['decision'] == 'no_default').sum()

    reporte = {
        'fecha':             datetime.now().strftime("%Y-%m-%d %H:%M"),
        'total_prestamos':   n_total,
        'aprobados':         int(n_aprobado),
        'rechazados':        int(n_default),
        'tasa_aprobacion':   round(n_aprobado / n_total, 4),
        'score_promedio':    round(resultados['score'].mean(), 2),
        'score_mediano':     round(resultados['score'].median(), 2),
        'distribucion_riesgo': {
            'bajo':  int((resultados['riesgo'] == 'bajo').sum()),
            'medio': int((resultados['riesgo'] == 'medio').sum()),
            'alto':  int((resultados['riesgo'] == 'alto').sum())
        }
    }

    ruta = REPORTS_DIR / 'batch_report_latest.json'
    with open(ruta, 'w') as f:
        json.dump(reporte, f, indent=2)

    logger.info("Reporte ejecutivo generado:")
    logger.info(f"  Total:       {n_total:,}")
    logger.info(f"  Aprobados:   {n_aprobado:,} ({n_aprobado/n_total:.1%})")
    logger.info(f"  Rechazados:  {n_default:,} ({n_default/n_total:.1%})")
    logger.info(f"  Score medio: {resultados['score'].mean():.0f}")


# ── Flow principal ─────────────────────────────────────────────────────────────

@flow(name="pipeline-batch-scoring-loanrisk")
def pipeline_batch_scoring():
    """
    Pipeline de scoring batch para el portfolio completo.

    Pasos:
    1. Cargar modelo en producción
    2. Cargar portfolio de préstamos
    3. Generar scores para todo el portfolio
    4. Guardar resultados en parquet
    5. Generar reporte ejecutivo en JSON
    """
    logger = get_run_logger()
    logger.info("Iniciando pipeline de batch scoring — LoanRisk-ML")

    model, preprocessor, threshold = cargar_modelo_produccion()
    df         = cargar_portfolio()
    resultados = generar_scores(model, preprocessor, threshold, df)
    guardar_resultados(resultados)
    generar_reporte_batch(resultados)

    logger.info("Pipeline de batch scoring completado")


if __name__ == "__main__":
    pipeline_batch_scoring.serve(
        name="loanrisk-batch-scoring-mensual",
        schedule={"cron": "0 10 1 * *"}  # día 1 de cada mes a las 10am
    )