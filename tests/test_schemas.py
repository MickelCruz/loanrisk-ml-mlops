import pytest
from pydantic import ValidationError
from api.schemas import LoanRequest


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
    }


def test_loan_request_valido(loan_valido):
    loan = LoanRequest(**loan_valido)
    assert loan.loan_amnt == 10000
    assert loan.grade == "B"


def test_emp_length_none_es_valido(loan_valido):
    loan_valido["emp_length"] = None
    loan = LoanRequest(**loan_valido)
    assert loan.emp_length is None


def test_fallback_home_ownership_invalido(loan_valido):
    loan_valido["home_ownership"] = "INVALIDO"
    loan = LoanRequest(**loan_valido)
    assert loan.home_ownership == "MORTGAGE"


def test_fallback_purpose_invalido(loan_valido):
    loan_valido["purpose"] = "comprar_carro_dominicano"
    loan = LoanRequest(**loan_valido)
    assert loan.purpose == "debt_consolidation"


def test_fallback_addr_state_invalido(loan_valido):
    loan_valido["addr_state"] = "XX"
    loan = LoanRequest(**loan_valido)
    assert loan.addr_state == "CA"


def test_loan_amnt_negativo_falla(loan_valido):
    loan_valido["loan_amnt"] = -5000
    with pytest.raises(ValidationError):
        LoanRequest(**loan_valido)


def test_annual_inc_cero_falla(loan_valido):
    loan_valido["annual_inc"] = 0
    with pytest.raises(ValidationError):
        LoanRequest(**loan_valido)


def test_grade_invalido_falla(loan_valido):
    loan_valido["grade"] = "Z"
    with pytest.raises(ValidationError):
        LoanRequest(**loan_valido)


def test_emp_length_invalido_falla(loan_valido):
    loan_valido["emp_length"] = "15 years"
    with pytest.raises(ValidationError):
        LoanRequest(**loan_valido)