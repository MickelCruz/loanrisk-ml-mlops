import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models'

# ── Cargar artefactos al iniciar la API ────────────────────────────────────────
model        = joblib.load(MODELS_DIR / 'XGBoost_best.joblib')
preprocessor = joblib.load(MODELS_DIR / 'preprocessor.joblib')
explainer    = joblib.load(MODELS_DIR / 'shap_explainer.joblib')

with open(MODELS_DIR / 'resultados_finales.json') as f:
    config = json.load(f)

with open(MODELS_DIR / 'valores_validos.json') as f:
    valores_validos = json.load(f)

THRESHOLD  = config['threshold']
SCORE_MIN  = config['scorecard']['score_min']
SCORE_MAX  = config['scorecard']['score_max']

# ── Caps calculados en 02_features.ipynb ──────────────────────────────────────
CAPS = {
    'payment_to_income':    0.193,
    'loan_to_income':       0.500,
    'revol_util_amount':    0.988,
    'total_debt_to_credit': 1.031
}

# ── Encodings definidos en 02_features.ipynb ──────────────────────────────────
GRADE_MAP     = {g: i for i, g in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'])}
SUB_GRADE_MAP = {f'{g}{n}': i for i, (g, n) in enumerate(
    [(g, n) for g in 'ABCDEFG' for n in range(1, 6)]
)}
EMP_LENGTH_MAP = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4,  '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8,  '9 years': 9, '10+ years': 10
}


def prob_a_score(prob: float) -> int:
    """Convierte probabilidad de default a score 300-850."""
    return int(SCORE_MAX - (SCORE_MAX - SCORE_MIN) * prob)


def clasificar_riesgo(score: int) -> str:
    """Clasifica el riesgo según el score."""
    if score >= 700:
        return 'bajo'
    elif score >= 600:
        return 'medio'
    return 'alto'


def transformar_datos(data: dict) -> np.ndarray:
    """
    Aplica el pipeline completo de transformación a un registro.
    Mismo orden que en 02_features.ipynb y 03_modeling.ipynb.
    """
    df = pd.DataFrame([data])

    # Ratio features
    df['payment_to_income']    = df['installment'] / (df['annual_inc'] / 12).replace(0, np.nan)
    df['loan_to_income']       = df['loan_amnt']   / df['annual_inc'].replace(0, np.nan)
    df['revol_util_amount']    = df['revol_bal']   / df['total_rev_hi_lim'].replace(0, np.nan)
    df['total_debt_to_credit'] = df['total_bal_ex_mort'] / df['tot_hi_cred_lim'].replace(0, np.nan)

    # Capping
    for col, cap in CAPS.items():
        df[col] = df[col].clip(upper=cap)

    # Features temporales
    df['issue_d']          = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
    df['issue_year']       = df['issue_d'].dt.year
    df['issue_month']      = df['issue_d'].dt.month
    df['credit_history_months'] = (
        (df['issue_d'].dt.year  - df['earliest_cr_line'].dt.year)  * 12 +
        (df['issue_d'].dt.month - df['earliest_cr_line'].dt.month)
    )
    df = df.drop(columns=['issue_d', 'earliest_cr_line'])

    # Encoding binario
    df['term']                 = df['term'].str.strip().map({'36 months': 0, '60 months': 1})
    df['initial_list_status']  = df['initial_list_status'].map({'w': 0, 'f': 1})
    df['application_type']     = df['application_type'].map({'Individual': 0, 'Joint App': 1})
    df['disbursement_method']  = df['disbursement_method'].map({'Cash': 0, 'DirectPay': 1})
    df['debt_settlement_flag'] = df['debt_settlement_flag'].map({'N': 0, 'Y': 1})

    # Encoding ordinal
    df['grade']      = df['grade'].map(GRADE_MAP)
    df['sub_grade']  = df['sub_grade'].map(SUB_GRADE_MAP)
    df['emp_length'] = df['emp_length'].map(EMP_LENGTH_MAP)

    # Features binarias para nulos informativos
    df['no_delinquency'] = df['mths_since_last_delinq'].isnull().astype(int)
    df['no_recent_inq']  = df['mths_since_recent_inq'].isnull().astype(int)

    # Preprocesador — imputación, scaling, TargetEncoder
    X = preprocessor.transform(df)

    return X


def predecir(data: dict) -> dict:
    """
    Genera predicción completa para un préstamo individual.
    """
    X     = transformar_datos(data)
    proba = float(model.predict_proba(X)[0][1])
    score = prob_a_score(proba)

    return {
        'probabilidad_default': round(proba, 4),
        'score':                score,
        'decision':             'default' if proba >= THRESHOLD else 'no_default',
        'riesgo':               clasificar_riesgo(score)
    }


def predecir_batch(registros: list) -> list:
    """
    Genera predicciones para múltiples préstamos.
    """
    return [predecir(r) for r in registros]


def explicar(data: dict) -> dict:
    """
    Genera explicación SHAP para un préstamo individual.
    """
    import shap

    df  = pd.DataFrame([data])
    X   = transformar_datos(data)

    # Recuperar nombres de features
    num_cols      = preprocessor.transformers_[0][2]
    cat_cols      = preprocessor.transformers_[1][2]
    feature_names = num_cols + cat_cols

    X_df = pd.DataFrame(X, columns=feature_names)

    shap_values  = explainer(X_df)
    contribuciones = dict(zip(
        feature_names,
        [round(float(v), 4) for v in shap_values.values[0]]
    ))

    # Top 5 features que más influyeron
    top_features = sorted(
        contribuciones.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    prediccion = predecir(data)

    return {
        **prediccion,
        'top_features': [
            {'feature': k, 'contribucion': v, 'direccion': 'aumenta_riesgo' if v > 0 else 'reduce_riesgo'}
            for k, v in top_features
        ]
    }