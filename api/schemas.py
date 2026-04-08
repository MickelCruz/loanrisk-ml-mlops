from pydantic import BaseModel, field_validator
from typing import Optional, List
import json
from pathlib import Path

# ── Cargar valores válidos ─────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
MODELS_DIR     = ROOT / 'models'

with open(MODELS_DIR / 'valores_validos.json') as f:
    valores_validos = json.load(f)

VALID_HOME_OWNERSHIP      = valores_validos['categoricos']['home_ownership']
VALID_VERIFICATION_STATUS = valores_validos['categoricos']['verification_status']
VALID_PURPOSE             = valores_validos['categoricos']['purpose']
VALID_ADDR_STATE          = valores_validos['categoricos']['addr_state']
VALID_GRADE               = valores_validos['ordinales']['grade']
VALID_EMP_LENGTH          = valores_validos['ordinales']['emp_length']

FALLBACKS = valores_validos['fallbacks']


# ── Schema de entrada ──────────────────────────────────────────────────────────
class LoanRequest(BaseModel):
    # Información del préstamo
    loan_amnt:    float
    term:         str
    int_rate:     float
    installment:  float
    grade:        str
    sub_grade:    str
    emp_length:   Optional[str] = None

    # Información del solicitante
    home_ownership:      str
    annual_inc:          float
    verification_status: str
    issue_d:             str
    purpose:             str
    addr_state:          str
    dti:                 Optional[float] = None
    earliest_cr_line:    str

    # Historial crediticio
    delinq_2yrs:            float = 0
    fico_range_low:         float = 670
    fico_range_high:        float = 674
    inq_last_6mths:         float = 0
    mths_since_last_delinq: Optional[float] = None
    open_acc:               float = 10
    pub_rec:                float = 0
    revol_bal:              float = 0
    revol_util:             Optional[float] = None
    total_acc:              float = 20

    # Variables adicionales
    initial_list_status:  str   = 'w'
    application_type:     str   = 'Individual'
    disbursement_method:  str   = 'Cash'
    debt_settlement_flag: str   = 'N'

    # Variables de buró
    tot_cur_bal:                Optional[float] = None
    total_rev_hi_lim:           Optional[float] = None
    acc_open_past_24mths:       Optional[float] = None
    avg_cur_bal:                Optional[float] = None
    bc_open_to_buy:             Optional[float] = None
    bc_util:                    Optional[float] = None
    mo_sin_old_il_acct:         Optional[float] = None
    mo_sin_old_rev_tl_op:       Optional[float] = None
    mo_sin_rcnt_rev_tl_op:      Optional[float] = None
    mo_sin_rcnt_tl:             Optional[float] = None
    mort_acc:                   Optional[float] = None
    mths_since_recent_bc:       Optional[float] = None
    mths_since_recent_inq:      Optional[float] = None
    num_accts_ever_120_pd:      Optional[float] = None
    num_actv_bc_tl:             Optional[float] = None
    num_actv_rev_tl:            Optional[float] = None
    num_bc_sats:                Optional[float] = None
    num_bc_tl:                  Optional[float] = None
    num_il_tl:                  Optional[float] = None
    num_op_rev_tl:              Optional[float] = None
    num_rev_accts:              Optional[float] = None
    num_rev_tl_bal_gt_0:        Optional[float] = None
    num_sats:                   Optional[float] = None
    num_tl_120dpd_2m:           Optional[float] = None
    num_tl_30dpd:               Optional[float] = None
    num_tl_90g_dpd_24m:         Optional[float] = None
    num_tl_op_past_12m:         Optional[float] = None
    pct_tl_nvr_dlq:             Optional[float] = None
    percent_bc_gt_75:           Optional[float] = None
    pub_rec_bankruptcies:       Optional[float] = None
    tax_liens:                  Optional[float] = None
    tot_hi_cred_lim:            Optional[float] = None
    total_bal_ex_mort:          Optional[float] = None
    total_bc_limit:             Optional[float] = None
    total_il_high_credit_limit: Optional[float] = None
    acc_now_delinq:             Optional[float] = None
    chargeoff_within_12_mths:   Optional[float] = None
    collections_12_mths_ex_med: Optional[float] = None
    delinq_amnt:                Optional[float] = None
    tot_coll_amt:               Optional[float] = None

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator('grade')
    @classmethod
    def grade_valido(cls, v):
        if v not in VALID_GRADE:
            raise ValueError(f"grade inválido: {v}. Valores válidos: {VALID_GRADE}")
        return v

    @field_validator('emp_length')
    @classmethod
    def emp_length_valido(cls, v):
        if v is not None and v not in VALID_EMP_LENGTH:
            raise ValueError(f"emp_length inválido: {v}")
        return v

    @field_validator('home_ownership')
    @classmethod
    def home_ownership_valido(cls, v):
        if v not in VALID_HOME_OWNERSHIP:
            return FALLBACKS['home_ownership']
        return v

    @field_validator('verification_status')
    @classmethod
    def verification_status_valido(cls, v):
        if v not in VALID_VERIFICATION_STATUS:
            return FALLBACKS['verification_status']
        return v

    @field_validator('purpose')
    @classmethod
    def purpose_valido(cls, v):
        if v not in VALID_PURPOSE:
            return FALLBACKS['purpose']
        return v

    @field_validator('addr_state')
    @classmethod
    def addr_state_valido(cls, v):
        if v not in VALID_ADDR_STATE:
            return FALLBACKS['addr_state']
        return v

    @field_validator('annual_inc')
    @classmethod
    def annual_inc_positivo(cls, v):
        if v <= 0:
            raise ValueError("annual_inc debe ser mayor a 0")
        return v

    @field_validator('loan_amnt')
    @classmethod
    def loan_amnt_valido(cls, v):
        if v <= 0:
            raise ValueError("loan_amnt debe ser mayor a 0")
        return v


# ── Schema de respuesta individual ────────────────────────────────────────────
class LoanResponse(BaseModel):
    probabilidad_default: float
    score:                int
    decision:             str
    riesgo:               str


# ── Schema de respuesta con explicación ───────────────────────────────────────
class FeatureContribucion(BaseModel):
    feature:      str
    contribucion: float
    direccion:    str


class LoanExplainResponse(BaseModel):
    probabilidad_default: float
    score:                int
    decision:             str
    riesgo:               str
    top_features:         List[FeatureContribucion]


# ── Schema para batch ──────────────────────────────────────────────────────────
class BatchRequest(BaseModel):
    prestamos: List[LoanRequest]


class BatchResponse(BaseModel):
    resultados: List[LoanResponse]
    total:      int