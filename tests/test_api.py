import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


@pytest.fixture
def payload_valido():
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
        "delinq_2yrs":          0.0,
        "inq_last_6mths":       0.0,
        "open_acc":             10.0,
        "pub_rec":              0.0,
        "revol_util":           25.0,
        "total_acc":            20.0,
        "mths_since_last_delinq":     None,
        "mths_since_recent_inq":      None,
        "tot_cur_bal":                None,
        "acc_open_past_24mths":       None,
        "avg_cur_bal":                None,
        "bc_open_to_buy":             None,
        "bc_util":                    None,
        "mo_sin_old_il_acct":         None,
        "mo_sin_old_rev_tl_op":       None,
        "mo_sin_rcnt_rev_tl_op":      None,
        "mo_sin_rcnt_tl":             None,
        "mort_acc":                   None,
        "mths_since_recent_bc":       None,
        "num_accts_ever_120_pd":      None,
        "num_actv_bc_tl":             None,
        "num_actv_rev_tl":            None,
        "num_bc_sats":                None,
        "num_bc_tl":                  None,
        "num_il_tl":                  None,
        "num_op_rev_tl":              None,
        "num_rev_accts":              None,
        "num_rev_tl_bal_gt_0":        None,
        "num_sats":                   None,
        "num_tl_120dpd_2m":           None,
        "num_tl_30dpd":               None,
        "num_tl_90g_dpd_24m":         None,
        "num_tl_op_past_12m":         None,
        "pct_tl_nvr_dlq":             None,
        "percent_bc_gt_75":           None,
        "pub_rec_bankruptcies":       None,
        "tax_liens":                  None,
        "total_bc_limit":             None,
        "total_il_high_credit_limit": None,
        "acc_now_delinq":             None,
        "chargeoff_within_12_mths":   None,
        "collections_12_mths_ex_med": None,
        "delinq_amnt":                None,
        "tot_coll_amt":               None,
    }


# ── Health ─────────────────────────────────────────────────────────────────────
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["modelo"] == "XGBoost"


# ── Model Info ─────────────────────────────────────────────────────────────────
def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "modelo"    in data
    assert "threshold" in data
    assert "metricas"  in data


# ── Score ──────────────────────────────────────────────────────────────────────
def test_score_status_code(payload_valido):
    response = client.post("/score", json=payload_valido)
    assert response.status_code == 200

def test_score_campos_respuesta(payload_valido):
    response = client.post("/score", json=payload_valido)
    data = response.json()
    assert "probabilidad_default" in data
    assert "score"                in data
    assert "decision"             in data
    assert "riesgo"               in data

def test_score_valores_validos(payload_valido):
    response = client.post("/score", json=payload_valido)
    data = response.json()
    assert 300 <= data["score"] <= 850
    assert 0.0 <= data["probabilidad_default"] <= 1.0
    assert data["decision"] in ["default", "no_default"]
    assert data["riesgo"]   in ["bajo", "medio", "alto"]

def test_score_payload_invalido():
    response = client.post("/score", json={"loan_amnt": -1000})
    assert response.status_code == 422


# ── Explain ────────────────────────────────────────────────────────────────────
def test_explain_status_code(payload_valido):
    response = client.post("/explain", json=payload_valido)
    assert response.status_code == 200

def test_explain_tiene_top_features(payload_valido):
    response = client.post("/explain", json=payload_valido)
    data = response.json()
    assert "top_features" in data
    assert len(data["top_features"]) == 5

def test_explain_direccion_valida(payload_valido):
    response = client.post("/explain", json=payload_valido)
    data = response.json()
    for feature in data["top_features"]:
        assert feature["direccion"] in ["aumenta_riesgo", "reduce_riesgo"]


# ── Batch ──────────────────────────────────────────────────────────────────────
def test_score_batch_status_code(payload_valido):
    response = client.post("/score/batch", json={"prestamos": [payload_valido, payload_valido]})
    assert response.status_code == 200

def test_score_batch_total_correcto(payload_valido):
    response = client.post("/score/batch", json={"prestamos": [payload_valido, payload_valido]})
    data = response.json()
    assert data["total"] == 2
    assert len(data["resultados"]) == 2