import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / 'data' / 'processed'
MODELS_DIR     = ROOT / 'models'
REPORTS_DIR    = ROOT / 'reports'

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def cargar_datos():
    """Carga los conjuntos de train y test con nombres de features."""
    X_train = pd.read_parquet(DATA_PROCESSED / 'X_train.parquet')
    X_test  = pd.read_parquet(DATA_PROCESSED / 'X_test.parquet')

    # Recuperar nombres de features del preprocesador
    preprocessor  = joblib.load(MODELS_DIR / 'preprocessor.joblib')
    num_cols      = preprocessor.transformers_[0][2]
    cat_cols      = preprocessor.transformers_[1][2]
    feature_names = num_cols + cat_cols

    X_train.columns = feature_names
    X_test.columns  = feature_names

    return X_train, X_test, feature_names


def seleccionar_top_features(feature_names: list, n: int = 15) -> list:
    """
    Selecciona las top features más importantes según el modelo.
    Usa las features del beeswarm plot de SHAP.
    """
    # Top features identificadas en 05_explainability.ipynb
    top_features = [
        'debt_settlement_flag',
        'int_rate',
        'term',
        'sub_grade',
        'dti',
        'acc_open_past_24mths',
        'grade',
        'home_ownership',
        'addr_state',
        'mo_sin_old_rev_tl_op',
        'loan_to_income',
        'loan_amnt',
        'total_bc_limit',
        'fico_range_low',
        'emp_length'
    ]

    # Filtrar solo las que existen en el dataset
    return [f for f in top_features if f in feature_names][:n]


def generar_reporte_drift(X_train, X_test, features: list) -> dict:
    """
    Genera reporte de drift entre train (referencia) y test (producción).
    """
    ref_sample  = X_train.sample(n=10000, random_state=42)[features]
    curr_sample = X_test[features]

    # Definir datasets de Evidently
    column_mapping = DataDefinition(numerical_columns=features)

    ref_dataset  = Dataset.from_pandas(ref_sample,  data_definition=column_mapping)
    curr_dataset = Dataset.from_pandas(curr_sample, data_definition=column_mapping)

    # Generar reporte
    report   = Report(metrics=[DataSummaryPreset(), DataDriftPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=curr_dataset)

    # Guardar HTML
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_html   = REPORTS_DIR / f'drift_report_{timestamp}.html'
    ruta_latest = REPORTS_DIR / 'drift_report_latest.html'

    snapshot.save_html(str(ruta_html))
    snapshot.save_html(str(ruta_latest))

    print(f"Reporte guardado en: {ruta_html}")

    return snapshot.dict()


def analizar_resultados(results: dict, features: list) -> dict:
    """
    Extrae métricas clave del reporte de drift.
    """
    metricas = {}

    try:
        for m in results.get('metrics', []):
            nombre = m.get('metric_name', '')
            valor  = m.get('value', None)
            if valor is not None:
                metricas[nombre] = valor
    except Exception:
        pass

    # Construir resumen
    resumen = {
        'fecha':              datetime.now().strftime("%Y-%m-%d %H:%M"),
        'features_analizadas': len(features),
        'metricas':           metricas
    }

    return resumen


def guardar_estado_drift(resumen: dict):
    """Guarda el estado del drift en JSON para consulta programática."""
    ruta = MODELS_DIR / 'drift_status.json'
    with open(ruta, 'w') as f:
        json.dump(resumen, f, indent=2)
    print(f"Estado de drift guardado en: {ruta}")


def run_drift_report():
    """
    Función principal — ejecuta el pipeline completo de drift detection.
    """
    print("Iniciando análisis de drift — LoanRisk-ML")
    print("=" * 50)

    # 1. Cargar datos
    print("Cargando datos...")
    X_train, X_test, feature_names = cargar_datos()
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # 2. Seleccionar top features
    features = seleccionar_top_features(feature_names)
    print(f"\nFeatures analizadas ({len(features)}):")
    for f in features:
        print(f"  - {f}")

    # 3. Generar reporte
    print("\nGenerando reporte de drift...")
    results = generar_reporte_drift(X_train, X_test, features)

    # 4. Analizar resultados
    resumen = analizar_resultados(results, features)

    # 5. Guardar estado
    guardar_estado_drift(resumen)

    print("\n" + "=" * 50)
    print("Análisis de drift completado")
    print(f"Features analizadas: {resumen['features_analizadas']}")
    print(f"Reporte HTML: reports/drift_report_latest.html")


if __name__ == "__main__":
    run_drift_report()