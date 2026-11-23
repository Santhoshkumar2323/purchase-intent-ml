import numpy as np
import pandas as pd

from src.drift.categorical import max_shift, shares
from src.drift.psi import make_edges, proportions, psi
from src.drift.report import compare

RNG = np.random.default_rng(0)
BASE = RNG.normal(0, 1, 5000)
EDGES = make_edges(BASE)


def _psi_against(values) -> float:
    return psi(proportions(BASE, EDGES), proportions(values, EDGES))


def test_same_distribution_has_small_psi():
    assert _psi_against(RNG.normal(0, 1, 5000)) < 0.05


def test_shifted_distribution_is_an_alert():
    assert _psi_against(RNG.normal(1.5, 1, 5000)) > 0.25


def test_constant_column_does_not_crash():
    edges = make_edges([3, 3, 3, 3])
    assert psi(proportions([3, 3], edges), proportions([3, 3, 3], edges)) < 0.01


def test_category_shift_finds_the_worst_category():
    before = shares(["a"] * 80 + ["b"] * 20)
    after = shares(["a"] * 50 + ["b"] * 50)
    shift, category = max_shift(before, after)
    assert abs(shift - 0.3) < 1e-6 and category in {"a", "b"}


def test_small_batch_returns_error_not_exception():
    result = compare(pd.DataFrame({"x": [1, 2, 3]}), {"numeric": {}, "categorical": {}})
    assert "error" in result