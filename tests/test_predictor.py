import pytest
from api.predictor import predecir, explicar, prob_a_score, clasificar_riesgo


@pytest.fixture
def loan_valido():
    return {
        "loan_amnt":           10000,
        "term":                "36 months",
        "int_rate":            11.5,
        "installment":         327.43,
        "grade":               "B",
        "sub_grade":           "B3",
        "emp_length":          "5 years",
        "home_ownership":      "RENT",
        "annual_inc":          60000,
        "verification_status": "Verified",
        "issue_d":             "Jan-2018",
        "purpose":             "debt_consolidation",
        "addr_state":          "CA",
        "earliest_cr_line":    "Jan-2010",
        "dti":                 15.0,
        "fico_range_low":      670,
        "fico_range_high":     674,
        "revol_bal":           5000,
        "total_rev_hi_lim":    20000,
        "total_bal_ex_mort":   8000,
        "tot_hi_cred_lim":     30000,
        "initial_list_status":  "w",
        "application_type":     "Individual",
        "disbursement_method":  "Cash",
        "debt_settlement_flag": "N",
        "delinq_2yrs":    0.0,
        "inq_last_6mths": 0.0,
        "open_acc":       10.0,
        "pub_rec":        0.0,
        "revol_util":     25.0,
        "total_acc":      20.0,
        "mths_since_last_delinq":       None,
        "mths_since_recent_inq":        None,
        "tot_cur_bal":                  None,
        "acc_open_past_24mths":         None,
        "avg_cur_bal":                  None,
        "bc_open_to_buy":               None,
        "bc_util":                      None,
        "mo_sin_old_il_acct":           None,
        "mo_sin_old_rev_tl_op":         None,
        "mo_sin_rcnt_rev_tl_op":        None,
        "mo_sin_rcnt_tl":               None,
        "mort_acc":                     None,
        "mths_since_recent_bc":         None,
        "num_accts_ever_120_pd":        None,
        "num_actv_bc_tl":               None,
        "num_actv_rev_tl":              None,
        "num_bc_sats":                  None,
        "num_bc_tl":                    None,
        "num_il_tl":                    None,
        "num_op_rev_tl":                None,
        "num_rev_accts":                None,
        "num_rev_tl_bal_gt_0":          None,
        "num_sats":                     None,
        "num_tl_120dpd_2m":             None,
        "num_tl_30dpd":                 None,
        "num_tl_90g_dpd_24m":           None,
        "num_tl_op_past_12m":           None,
        "pct_tl_nvr_dlq":               None,
        "percent_bc_gt_75":             None,
        "pub_rec_bankruptcies":         None,
        "tax_liens":                    None,
        "total_bc_limit":               None,
        "total_il_high_credit_limit":   None,
        "acc_now_delinq":               None,
        "chargeoff_within_12_mths":     None,
        "collections_12_mths_ex_med":   None,
        "delinq_amnt":                  None,
        "tot_coll_amt":                 None,
    }


# ── Funciones auxiliares ───────────────────────────────────────────────────────
def test_prob_a_score_extremos():
    assert prob_a_score(0.0) == 850
    assert prob_a_score(1.0) == 300

def test_prob_a_score_medio():
    assert prob_a_score(0.5) == 575

def test_clasificar_riesgo_bajo():
    assert clasificar_riesgo(750) == "bajo"

def test_clasificar_riesgo_medio():
    assert clasificar_riesgo(650) == "medio"

def test_clasificar_riesgo_alto():
    assert clasificar_riesgo(500) == "alto"


# ── Predicción ─────────────────────────────────────────────────────────────────
def test_predecir_retorna_campos_correctos(loan_valido):
    resultado = predecir(loan_valido)
    assert "probabilidad_default" in resultado
    assert "score"                in resultado
    assert "decision"             in resultado
    assert "riesgo"               in resultado

def test_predecir_score_en_rango(loan_valido):
    resultado = predecir(loan_valido)
    assert 300 <= resultado["score"] <= 850

def test_predecir_probabilidad_en_rango(loan_valido):
    resultado = predecir(loan_valido)
    assert 0.0 <= resultado["probabilidad_default"] <= 1.0

def test_predecir_decision_valida(loan_valido):
    resultado = predecir(loan_valido)
    assert resultado["decision"] in ["default", "no_default"]

def test_predecir_riesgo_valido(loan_valido):
    resultado = predecir(loan_valido)
    assert resultado["riesgo"] in ["bajo", "medio", "alto"]


# ── Explicabilidad ─────────────────────────────────────────────────────────────
def test_explicar_retorna_campos_correctos(loan_valido):
    resultado = explicar(loan_valido)
    assert "probabilidad_default" in resultado
    assert "score"                in resultado
    assert "decision"             in resultado
    assert "riesgo"               in resultado
    assert "top_features"         in resultado

def test_explicar_top_features_longitud(loan_valido):
    resultado = explicar(loan_valido)
    assert len(resultado["top_features"]) == 5

def test_explicar_direccion_valida(loan_valido):
    resultado = explicar(loan_valido)
    assert resultado["top_features"][0]["direccion"] in ["aumenta_riesgo", "reduce_riesgo"]