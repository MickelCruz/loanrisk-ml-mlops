import pandas as pd
import numpy as np
import joblib
import json
import mlflow
from pathlib import Path
from datetime import datetime

from prefect import task, flow, get_run_logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import xgboost as xgb

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / 'data' / 'processed'
MODELS_DIR     = ROOT / 'models'
MLFLOW_DIR     = ROOT / 'mlruns'


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="verificar_drift")
def verificar_drift():
    """
    Consulta drift_status.json antes de reentrenar.
    Loguea el estado del drift para trazabilidad.
    """
    logger = get_run_logger()

    ruta = MODELS_DIR / 'drift_status.json'

    if not ruta.exists():
        logger.info("drift_status.json no encontrado — procediendo sin verificación")
        return

    with open(ruta) as f:
        status = json.load(f)

    alerta  = status.get('alerta', False)
    pct     = status.get('pct_drift', 0)
    fecha   = status.get('fecha', 'desconocida')
    n_drift = status.get('features_con_drift', 0)
    total   = status.get('features_analizadas', 0)

    logger.info(f"Estado de drift al {fecha}:")
    logger.info(f"  Features con drift: {n_drift}/{total} ({pct:.0%})")

    if alerta:
        logger.info("✅ Drift detectado — reentrenamiento justificado")
        for f in status.get('features_afectadas', []):
            logger.info(f"  - {f}")
    else:
        logger.info("ℹ️ Sin drift significativo — reentrenando por schedule mensual")


@task(name="cargar_datos")
def cargar_datos() -> pd.DataFrame:
    """Carga el dataset de features procesadas."""
    logger = get_run_logger()
    df = pd.read_parquet(DATA_PROCESSED / 'loan_features.parquet')
    logger.info(f"Dataset cargado: {df.shape}")
    logger.info(f"Default rate: {df['target'].mean():.2%}")
    return df


@task(name="split_datos")
def split_datos(df: pd.DataFrame):
    """Divide el dataset en train, val y test."""
    logger = get_run_logger()

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    logger.info(f"Train: {X_train.shape} — Val: {X_val.shape} — Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


@task(name="construir_preprocesador")
def construir_preprocesador(X_train, y_train):
    """Construye y ajusta el ColumnTransformer."""
    logger = get_run_logger()

    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('target_encoder', TargetEncoder(cols=cat_cols))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    preprocessor.fit(X_train, y_train)
    logger.info("Preprocesador ajustado correctamente")
    return preprocessor


@task(name="transformar_datos")
def transformar_datos(preprocessor, X_train, X_val, X_test):
    """Transforma los conjuntos con el preprocesador ajustado."""
    logger = get_run_logger()

    X_train_prep = preprocessor.transform(X_train)
    X_val_prep   = preprocessor.transform(X_val)
    X_test_prep  = preprocessor.transform(X_test)

    logger.info(f"Datos transformados — Train: {X_train_prep.shape}")
    return X_train_prep, X_val_prep, X_test_prep


@task(name="entrenar_modelo")
def entrenar_modelo(X_train_prep, y_train, y_val):
    """Entrena XGBoost con los mejores hiperparámetros guardados."""
    logger = get_run_logger()

    with open(MODELS_DIR / 'best_params.json') as f:
        best_params = json.load(f)

    params = best_params['XGBoost'].copy()
    params['random_state']      = 42
    params['n_jobs']            = -1
    params['verbosity']         = 0
    params['eval_metric']       = 'logloss'
    params['scale_pos_weight']  = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_prep, y_train)

    logger.info(f"Modelo entrenado con scale_pos_weight: {params['scale_pos_weight']:.4f}")
    return model


@task(name="evaluar_modelo")
def evaluar_modelo(model, X_val_prep, y_val, X_test_prep, y_test):
    """Evalúa el modelo y retorna métricas."""
    logger = get_run_logger()

    p_val  = model.predict_proba(X_val_prep)[:, 1]
    p_test = model.predict_proba(X_test_prep)[:, 1]

    auc_pr_val  = average_precision_score(y_val,  p_val)
    auc_pr_test = average_precision_score(y_test, p_test)
    auc_roc_val = roc_auc_score(y_val, p_val)

    fpr, tpr, _ = roc_curve(y_val, p_val)
    ks   = float((tpr - fpr).max())
    gini = float(2 * auc_roc_val - 1)

    metricas = {
        'auc_pr_val':  round(auc_pr_val, 4),
        'auc_pr_test': round(auc_pr_test, 4),
        'auc_roc_val': round(auc_roc_val, 4),
        'ks':          round(ks, 4),
        'gini':        round(gini, 4)
    }

    logger.info(f"AUC-PR Val: {auc_pr_val:.4f} | KS: {ks:.4f} | Gini: {gini:.4f}")
    return metricas


@task(name="comparar_con_produccion")
def comparar_con_produccion(metricas_nuevo: dict) -> bool:
    """
    Compara el nuevo modelo con el modelo en producción.
    Retorna True si el nuevo modelo es mejor.
    """
    logger = get_run_logger()

    with open(MODELS_DIR / 'resultados_finales.json') as f:
        config_actual = json.load(f)

    auc_pr_actual = config_actual['metricas']['auc_pr_val']
    auc_pr_nuevo  = metricas_nuevo['auc_pr_val']

    logger.info(f"AUC-PR actual en producción: {auc_pr_actual:.4f}")
    logger.info(f"AUC-PR nuevo modelo:         {auc_pr_nuevo:.4f}")

    es_mejor = auc_pr_nuevo > auc_pr_actual

    if es_mejor:
        logger.info("✅ El nuevo modelo es mejor — se procederá a reemplazar")
    else:
        logger.info("⚠️ El nuevo modelo no mejora al actual — se mantiene en producción")

    return es_mejor


@task(name="guardar_modelo")
def guardar_modelo(model, preprocessor, metricas: dict):
    """Guarda el nuevo modelo y actualiza la configuración."""
    logger = get_run_logger()

    joblib.dump(model,        MODELS_DIR / 'XGBoost_best.joblib')
    joblib.dump(preprocessor, MODELS_DIR / 'preprocessor.joblib')

    with open(MODELS_DIR / 'resultados_finales.json') as f:
        config = json.load(f)

    config['metricas']             = metricas
    config['ultima_actualizacion'] = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(MODELS_DIR / 'resultados_finales.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Modelo y configuración actualizados correctamente")


# ── Flow principal ─────────────────────────────────────────────────────────────

@flow(name="pipeline-reentrenamiento-loanrisk")
def pipeline_reentrenamiento():
    """
    Pipeline completo de reentrenamiento del modelo LoanRisk-ML.

    Pasos:
    0. Verificar estado de drift
    1. Cargar datos de features procesadas
    2. Split train/val/test
    3. Construir y ajustar preprocesador
    4. Transformar datos
    5. Entrenar modelo con mejores hiperparámetros
    6. Evaluar métricas
    7. Comparar con modelo en producción
    8. Si es mejor → guardar y actualizar configuración
    """
    logger = get_run_logger()
    logger.info("Iniciando pipeline de reentrenamiento — LoanRisk-ML")

    # Paso 0 — Verificar drift
    verificar_drift()

    # Paso 1 — Cargar datos
    df = cargar_datos()

    # Paso 2 — Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_datos(df)

    # Paso 3 — Preprocesador
    preprocessor = construir_preprocesador(X_train, y_train)

    # Paso 4 — Transformar
    X_train_prep, X_val_prep, X_test_prep = transformar_datos(
        preprocessor, X_train, X_val, X_test
    )

    # Paso 5 — Entrenar
    model = entrenar_modelo(X_train_prep, y_train, y_val)

    # Paso 6 — Evaluar
    metricas = evaluar_modelo(model, X_val_prep, y_val, X_test_prep, y_test)

    # Paso 7 — Comparar con producción
    es_mejor = comparar_con_produccion(metricas)

    # Paso 8 — Guardar si es mejor
    if es_mejor:
        guardar_modelo(model, preprocessor, metricas)
        logger.info("Pipeline completado — nuevo modelo en producción")
    else:
        logger.info("Pipeline completado — modelo anterior se mantiene")


if __name__ == "__main__":
    pipeline_reentrenamiento.serve(
        name="loanrisk-reentrenamiento-mensual",
        schedule={"cron": "0 8 1 * *"}  # día 1 de cada mes a las 8am
    )