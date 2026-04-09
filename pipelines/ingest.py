import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from prefect import task, flow, get_run_logger

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT / 'data' / 'raw'
DATA_PROCESSED = ROOT / 'data' / 'processed'


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="cargar_datos_crudos")
def cargar_datos_crudos() -> pd.DataFrame:
    """
    Carga el dataset crudo de Lending Club.
    En producción real este sería un archivo nuevo que llega periódicamente.
    """
    logger = get_run_logger()

    ruta = DATA_RAW / 'accepted_2007_to_2018Q4.csv'
    df   = pd.read_csv(ruta, low_memory=False)

    logger.info(f"Dataset crudo cargado: {df.shape}")
    return df


@task(name="definir_target")
def definir_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra préstamos con resultado definitivo y crea el target binario.
    """
    logger = get_run_logger()

    status_map = {'Fully Paid': 0, 'Charged Off': 1}
    df_model   = df[df['loan_status'].isin(status_map.keys())].copy()
    df_model['target'] = df_model['loan_status'].map(status_map)

    logger.info(f"Filas después del filtro: {len(df_model):,}")
    logger.info(f"Default rate: {df_model['target'].mean():.2%}")

    return df_model


@task(name="eliminar_columnas_nulos")
def eliminar_columnas_nulos(df: pd.DataFrame,
                             umbral: float = 0.50) -> pd.DataFrame:
    """
    Elimina columnas con más del umbral% de valores nulos.
    """
    logger = get_run_logger()

    missing_pct   = df.isnull().sum() / len(df)
    cols_to_drop  = missing_pct[missing_pct > umbral].index.tolist()
    df            = df.drop(columns=cols_to_drop)

    logger.info(f"Columnas eliminadas por nulos: {len(cols_to_drop)}")
    logger.info(f"Columnas restantes: {df.shape[1]}")

    return df


@task(name="eliminar_leakage")
def eliminar_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas con data leakage — información posterior al préstamo.
    """
    logger = get_run_logger()

    leakage_cols = [
        'loan_status', 'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
        'total_rec_int', 'total_rec_late_fee', 'recoveries',
        'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
        'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low',
        'url', 'id'
    ]

    leakage_cols = [col for col in leakage_cols if col in df.columns]
    df           = df.drop(columns=leakage_cols)

    logger.info(f"Columnas de leakage eliminadas: {len(leakage_cols)}")

    return df


@task(name="eliminar_baja_utilidad")
def eliminar_baja_utilidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas de baja utilidad predictiva.
    """
    logger = get_run_logger()

    cols_to_remove = [
        'pymnt_plan', 'hardship_flag', 'emp_title',
        'zip_code', 'title', 'funded_amnt',
        'funded_amnt_inv', 'policy_code'
    ]

    cols_to_remove = [col for col in cols_to_remove if col in df.columns]
    df             = df.drop(columns=cols_to_remove)

    logger.info(f"Columnas de baja utilidad eliminadas: {len(cols_to_remove)}")

    return df


@task(name="tratar_valores_centinela")
def tratar_valores_centinela(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza valores centinela con NaN.
    """
    logger = get_run_logger()

    centinelas = {
        'dti':              999,
        'tot_hi_cred_lim':  9999999,
        'total_rev_hi_lim': 9999999
    }

    for col, valor in centinelas.items():
        if col in df.columns:
            n = (df[col] == valor).sum()
            if n > 0:
                df.loc[df[col] == valor, col] = np.nan
                logger.info(f"{col}: {n} valores centinela reemplazados")

    return df


@task(name="crear_features")
def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de feature engineering.
    """
    logger = get_run_logger()

    # Ratio features
    df['payment_to_income']    = df['installment'] / (df['annual_inc'] / 12).replace(0, np.nan)
    df['loan_to_income']       = df['loan_amnt']   / df['annual_inc'].replace(0, np.nan)
    df['revol_util_amount']    = df['revol_bal']   / df['total_rev_hi_lim'].replace(0, np.nan)
    df['total_debt_to_credit'] = df['total_bal_ex_mort'] / df['tot_hi_cred_lim'].replace(0, np.nan)

    # Capping
    caps = {
        'payment_to_income':    0.193,
        'loan_to_income':       0.500,
        'revol_util_amount':    0.988,
        'total_debt_to_credit': 1.031
    }
    for col, cap in caps.items():
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
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df['grade'] = df['grade'].map({g: i for i, g in enumerate(grade_order)})

    sub_grades = [f'{g}{n}' for g in 'ABCDEFG' for n in range(1, 6)]
    df['sub_grade'] = df['sub_grade'].map({s: i for i, s in enumerate(sub_grades)})

    emp_length_order = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4,  '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8,  '9 years': 9, '10+ years': 10
    }
    df['emp_length'] = df['emp_length'].map(emp_length_order)

    # Features binarias para nulos informativos
    df['no_delinquency'] = df['mths_since_last_delinq'].isnull().astype(int)
    df['no_recent_inq']  = df['mths_since_recent_inq'].isnull().astype(int)

    # Imputación con mediana
    num_cols = df.select_dtypes(include='number').columns
    num_cols = [col for col in num_cols if col != 'target']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    logger.info(f"Features creadas — Shape final: {df.shape}")
    logger.info(f"Nulos restantes: {df.isnull().sum().sum()}")

    return df


@task(name="guardar_features")
def guardar_features(df: pd.DataFrame):
    """Guarda el dataset de features procesadas."""
    logger = get_run_logger()

    ruta = DATA_PROCESSED / 'loan_features.parquet'
    df.to_parquet(ruta, index=False)

    logger.info(f"Dataset guardado: {ruta}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Default rate: {df['target'].mean():.2%}")


# ── Flow principal ─────────────────────────────────────────────────────────────

@flow(name="pipeline-ingesta-loanrisk")
def pipeline_ingesta():
    """
    Pipeline completo de ingesta y feature engineering.

    Pasos:
    1. Cargar datos crudos
    2. Definir target binario
    3. Eliminar columnas con muchos nulos
    4. Eliminar data leakage
    5. Eliminar columnas de baja utilidad
    6. Tratar valores centinela
    7. Crear features
    8. Guardar dataset procesado
    """
    logger = get_run_logger()
    logger.info("Iniciando pipeline de ingesta — LoanRisk-ML")

    # Paso 1
    df = cargar_datos_crudos()

    # Paso 2
    df = definir_target(df)

    # Paso 3
    df = eliminar_columnas_nulos(df)

    # Paso 4
    df = eliminar_leakage(df)

    # Paso 5
    df = eliminar_baja_utilidad(df)

    # Paso 6
    df = tratar_valores_centinela(df)

    # Paso 7
    df = crear_features(df)

    # Paso 8
    guardar_features(df)

    logger.info("Pipeline de ingesta completado")


if __name__ == "__main__":
    pipeline_ingesta.serve(
        name="loanrisk-ingesta-mensual",
        schedule={"cron": "0 6 1 * *"}  # día 1 de cada mes a las 6am
    )