import pytest

from src.security.resolver import (
    AmbiguousCompanyError,
    CompanyNotResolvedError,
    resolve,
)


def test_resolve_ticker_in_training_universe():
    result = resolve("TCS")
    assert result.symbol == "TCS"
    assert result.in_training_universe is True
    assert result.resolved_by == "ticker"
    assert result.name == "Tata Consultancy Services"


def test_resolve_strips_ns_suffix_and_normalizes_case():
    result = resolve("tcs.NS")
    assert result.symbol == "TCS"
    assert result.in_training_universe is True


def test_resolve_by_company_name_alias():
    result = resolve("Tata Consultancy Services")
    assert result.symbol == "TCS"
    assert result.resolved_by == "name_alias"
    assert result.in_training_universe is True


def test_resolve_unseen_company_is_not_in_training_universe():
    result = resolve("DIXON")
    assert result.symbol == "DIXON"
    assert result.in_training_universe is False
    assert result.resolved_by == "unresolved_ticker_guess"
    assert result.name is None


def test_resolve_ambiguous_name_fragment_raises():
    with pytest.raises(AmbiguousCompanyError):
        resolve("Bajaj")


def test_resolve_empty_identifier_raises():
    with pytest.raises(CompanyNotResolvedError):
        resolve("   ")


def test_resolve_invalid_ticker_characters_raise():
    with pytest.raises(CompanyNotResolvedError):
        resolve("###not a ticker###")
