from src.config import ORIGINAL_UNIVERSE, UNIVERSE


def test_universe_has_50_unique_symbols():
    assert len(UNIVERSE) == 50
    assert len(set(UNIVERSE)) == 50


def test_original_5_stock_universe_is_a_subset_of_the_expanded_universe():
    assert set(ORIGINAL_UNIVERSE).issubset(set(UNIVERSE))
