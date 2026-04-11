import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

from prefect import task, flow, get_run_logger

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / 'data' / 'processed'
MODELS_DIR     = ROOT / 'models'
REPORTS_DIR    = ROOT / 'reports'

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Top features identificadas en 05_explainability.ipynb
TOP_FEATURES = [
    'debt_settlement_flag', 'int_rate', 'term', 'sub_grade', 'dti',
    'acc_open_past_24mths', 'grade', 'home_ownership', 'addr_state',
    'mo_sin_old_rev_tl_op', 'loan_to_income', 'loan_amnt',
    'total_bc_limit', 'fico_range_low', 'emp_length'
]


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="cargar_datos_drift")
def cargar_datos() -> tuple:
    """Carga los conjuntos de referencia y producción."""
    logger = get_run_logger()

    X_train = pd.read_parquet(DATA_PROCESSED / 'X_train.parquet')
    X_test  = pd.read_parquet(DATA_PROCESSED / 'X_test.parquet')

    preprocessor  = joblib.load(MODELS_DIR / 'preprocessor.joblib')
    num_cols      = preprocessor.transformers_[0][2]
    cat_cols      = preprocessor.transformers_[1][2]
    feature_names = num_cols + cat_cols

    X_train.columns = feature_names
    X_test.columns  = feature_names

    features = [f for f in TOP_FEATURES if f in feature_names]

    logger.info(f"Referencia: {X_train.shape} | Producción: {X_test.shape}")
    logger.info(f"Features a monitorear: {len(features)}")

    return X_train, X_test, features


@task(name="generar_reporte_drift")
def generar_reporte(X_train, X_test, features: list) -> dict:
    """Genera el reporte de drift con Evidently."""
    logger = get_run_logger()

    ref_sample  = X_train.sample(n=10000, random_state=42)[features]
    curr_sample = X_test[features]

    column_mapping = DataDefinition(numerical_columns=features)
    ref_dataset    = Dataset.from_pandas(ref_sample,  data_definition=column_mapping)
    curr_dataset   = Dataset.from_pandas(curr_sample, data_definition=column_mapping)

    report   = Report(metrics=[DataSummaryPreset(), DataDriftPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=curr_dataset)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_html   = REPORTS_DIR / f'drift_report_{timestamp}.html'
    ruta_latest = REPORTS_DIR / 'drift_report_latest.html'

    snapshot.save_html(str(ruta_html))
    snapshot.save_html(str(ruta_latest))

    logger.info(f"Reporte guardado: {ruta_html}")

    return snapshot.dict()


@task(name="evaluar_drift")
def evaluar_drift(results: dict, features: list) -> dict:
    """
    Evalúa si hay drift y construye el resumen.
    Umbral: si más del 30% de features tienen drift → alerta.
    """
    logger = get_run_logger()

    metricas = {}
    features_con_drift = []

    try:
        for m in results.get('metrics', []):
            nombre = m.get('metric_name', '')
            valor  = m.get('value', None)
            if valor is not None:
                metricas[nombre] = valor

            # Detectar features con drift
            if 'ValueDrift' in nombre and isinstance(valor, float):
                if valor > 0.1:
                    feature = nombre.split('column=')[1].split(',')[0] if 'column=' in nombre else ''
                    if feature:
                        features_con_drift.append(feature)
    except Exception:
        pass

    n_drift      = len(features_con_drift)
    pct_drift    = n_drift / len(features) if features else 0
    hay_alerta   = pct_drift > 0.30

    if hay_alerta:
        logger.warning(f"⚠️ ALERTA DE DRIFT — {n_drift}/{len(features)} features con drift ({pct_drift:.0%})")
        for f in features_con_drift:
            logger.warning(f"  - {f}")
    else:
        logger.info(f"✅ Sin drift significativo — {n_drift}/{len(features)} features afectadas")

    resumen = {
        'fecha':               datetime.now().strftime("%Y-%m-%d %H:%M"),
        'features_analizadas': len(features),
        'features_con_drift':  n_drift,
        'pct_drift':           round(pct_drift, 4),
        'alerta':              hay_alerta,
        'features_afectadas':  features_con_drift,
        'metricas':            metricas
    }

    return resumen


@task(name="guardar_estado_drift")
def guardar_estado(resumen: dict):
    """Persiste el estado del drift en JSON."""
    logger = get_run_logger()

    ruta = MODELS_DIR / 'drift_status.json'
    with open(ruta, 'w') as f:
        json.dump(resumen, f, indent=2)

    logger.info(f"Estado guardado: {ruta}")
    logger.info(f"Alerta activa: {resumen['alerta']}")


# ── Flow principal ─────────────────────────────────────────────────────────────

@flow(name="pipeline-drift-monitor-loanrisk")
def pipeline_drift_monitor():
    """
    Pipeline de monitoreo de drift — LoanRisk-ML.

    Pasos:
    1. Cargar datos de referencia (train) y producción (test)
    2. Generar reporte Evidently con las top 15 features
    3. Evaluar si hay drift significativo (umbral 30%)
    4. Guardar estado en drift_status.json
    5. Alertar si hay drift
    """
    logger = get_run_logger()
    logger.info("Iniciando monitoreo de drift — LoanRisk-ML")

    # Paso 1 — Cargar datos
    X_train, X_test, features = cargar_datos()

    # Paso 2 — Generar reporte
    results = generar_reporte(X_train, X_test, features)

    # Paso 3 — Evaluar drift
    resumen = evaluar_drift(results, features)

    # Paso 4 — Guardar estado
    guardar_estado(resumen)

    # Paso 5 — Log final
    if resumen['alerta']:
        logger.warning("Pipeline completado con ALERTA DE DRIFT")
        logger.warning("Considerar reentrenamiento del modelo")
    else:
        logger.info("Pipeline completado — modelo estable")


if __name__ == "__main__":
    pipeline_drift_monitor.serve(
        name="loanrisk-drift-monitor-semanal",
        schedule={"cron": "0 7 1 * *"}  # día 1 de cada mes a las 7am
    )