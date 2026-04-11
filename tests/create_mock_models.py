# tests/create_mock_models.py
# ══════════════════════════════════════════════════════════════════════════════
# Crea modelos mock para el entorno de CI/CD.
# Los modelos reales (.joblib) no se suben a GitHub por su tamaño.
# Este script genera versiones ligeras con la misma estructura.
# ══════════════════════════════════════════════════════════════════════════════

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
import shap

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Columnas — misma estructura que el preprocesador real ──────────────────────
NUM_COLS = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
    'emp_length', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
    'application_type', 'disbursement_method', 'debt_settlement_flag',
    'tot_cur_bal', 'total_rev_hi_lim', 'acc_open_past_24mths', 'avg_cur_bal',
    'bc_open_to_buy', 'bc_util', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc',
    'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl',
    'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
    'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
    'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq',
    'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
    'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
    'acc_now_delinq', 'chargeoff_within_12_mths', 'collections_12_mths_ex_med',
    'delinq_amnt', 'tot_coll_amt', 'payment_to_income', 'loan_to_income',
    'revol_util_amount', 'total_debt_to_credit', 'issue_year', 'issue_month',
    'credit_history_months', 'no_delinquency', 'no_recent_inq'
]

CAT_COLS = ['home_ownership', 'verification_status', 'purpose', 'addr_state']

ALL_COLS = NUM_COLS + CAT_COLS
N_FEATURES = len(ALL_COLS)

print(f"Creando modelos mock con {N_FEATURES} features...")

# ── Datos sintéticos para fitting ──────────────────────────────────────────────
np.random.seed(42)
N = 200

X_num = np.random.randn(N, len(NUM_COLS))
X_cat = np.random.choice(
    ['MORTGAGE', 'RENT', 'OWN', 'ANY'], size=(N, len(CAT_COLS))
)

import pandas as pd
X = pd.DataFrame(
    np.hstack([X_num, X_cat]),
    columns=ALL_COLS
)
for col in NUM_COLS:
    X[col] = pd.to_numeric(X[col])

y = np.random.randint(0, 2, N)

# ── Preprocesador mock ─────────────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('target_encoder', TargetEncoder(cols=CAT_COLS))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, NUM_COLS),
    ('cat', categorical_transformer, CAT_COLS)
])

preprocessor.fit(X, y)
X_prep = preprocessor.transform(X)

# ── Modelo XGBoost mock ────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=5,
    max_depth=2,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
model.fit(X_prep, y)

# ── SHAP Explainer mock ────────────────────────────────────────────────────────
explainer = shap.TreeExplainer(model)

# ── Guardar artefactos ─────────────────────────────────────────────────────────
joblib.dump(model,        MODELS_DIR / 'XGBoost_best.joblib')
joblib.dump(model,        MODELS_DIR / 'LightGBM_best.joblib')
joblib.dump(preprocessor, MODELS_DIR / 'preprocessor.joblib')
joblib.dump(explainer,    MODELS_DIR / 'shap_explainer.joblib')

print("Modelos mock guardados.")

# ── resultados_finales.json ────────────────────────────────────────────────────
resultados = {
    "modelo":    "XGBoost",
    "threshold": 0.5387,
    "metricas": {
        "auc_pr_val":    0.565,
        "auc_pr_test":   0.570,
        "auc_roc_val":   0.787,
        "ks":            0.417,
        "gini":          0.573,
        "f1_optimo":     0.508,
        "gap_train_val": 0.059
    },
    "scorecard": {
        "score_min":             300,
        "score_max":             850,
        "score_medio_no_default": 650,
        "score_medio_default":    511
    },
    "escala_riesgo": {
        "bajo":  "score >= 700",
        "medio": "score >= 600 y < 700",
        "alto":  "score < 600"
    }
}

with open(MODELS_DIR / 'resultados_finales.json', 'w') as f:
    json.dump(resultados, f, indent=2)

# ── valores_validos.json ───────────────────────────────────────────────────────
valores_validos = {
    "ordinales": {
        "grade":      ["A","B","C","D","E","F","G"],
        "sub_grade":  [f"{g}{n}" for g in "ABCDEFG" for n in range(1,6)],
        "emp_length": ["< 1 year","1 year","2 years","3 years","4 years",
                       "5 years","6 years","7 years","8 years","9 years","10+ years"]
    },
    "categoricos": {
        "home_ownership":      ["MORTGAGE","RENT","OWN","ANY"],
        "verification_status": ["Not Verified","Source Verified","Verified"],
        "purpose":             ["debt_consolidation","small_business","home_improvement",
                                "major_purchase","credit_card","other","house","vacation",
                                "car","medical","moving","renewable_energy","wedding","educational"],
        "addr_state":          ["PA","SD","IL","GA","MN","SC","RI","NC","CA","VA","AZ",
                                "IN","MD","NY","TX","KS","NM","AL","WA","OH","LA","FL",
                                "CO","MI","MO","DC","MA","WI","HI","VT","NJ","DE","TN",
                                "NH","NE","OR","CT","AR","NV","WV","MT","WY","OK","KY",
                                "MS","UT","ND","ME","AK","ID"]
    },
    "fallbacks": {
        "home_ownership":      "MORTGAGE",
        "verification_status": "Source Verified",
        "purpose":             "debt_consolidation",
        "addr_state":          "CA",
        "term":                0,
        "initial_list_status": 0,
        "application_type":    0,
        "disbursement_method": 0,
        "debt_settlement_flag": 0
    }
}

with open(MODELS_DIR / 'valores_validos.json', 'w') as f:
    json.dump(valores_validos, f, indent=2)

# ── best_params.json ───────────────────────────────────────────────────────────
best_params = {
    "XGBoost":  {"n_estimators": 5, "max_depth": 2},
    "LightGBM": {"n_estimators": 5, "num_leaves": 4}
}

with open(MODELS_DIR / 'best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print("JSONs de configuración guardados.")
print("✅ Modelos mock listos para CI/CD")