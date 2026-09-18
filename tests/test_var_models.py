"""
Unit tests for VaR models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from risk.var_models import ValueAtRisk


def approx_equal(a, b, tol):
    return abs(a - b) < tol


def _make_synthetic_returns(n_days=2000, n_assets=3, seed=1):
    rng = np.random.default_rng(seed)
    corr = np.full((n_assets, n_assets), 0.2)
    np.fill_diagonal(corr, 1.0)
    vols = np.array([0.01, 0.015, 0.012][:n_assets])
    cov = np.outer(vols, vols) * corr
    returns = pd.DataFrame(
        rng.multivariate_normal(mean=[0.0003] * n_assets, cov=cov, size=n_days),
        columns=[f"A{i}" for i in range(n_assets)]
    )
    return returns


def test_weights_must_sum_to_one():
    returns = _make_synthetic_returns(n_assets=3)
    try:
        ValueAtRisk(returns, weights=np.array([0.5, 0.5, 0.5]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_weights_length_must_match_assets():
    returns = _make_synthetic_returns(n_assets=3)
    try:
        ValueAtRisk(returns, weights=np.array([0.5, 0.5]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_var_is_positive_for_normal_returns():
    returns = _make_synthetic_returns()
    weights = np.array([0.4, 0.3, 0.3])
    var_model = ValueAtRisk(returns, weights)

    assert var_model.historical_var() > 0
    assert var_model.parametric_var() > 0
    assert var_model.monte_carlo_var() > 0


def test_all_three_methods_agree_on_normal_data():
    # When the underlying data really is normal, all three approaches
    # should converge to a similar answer (within a reasonable tolerance)
    returns = _make_synthetic_returns(n_days=5000)
    weights = np.array([0.4, 0.3, 0.3])
    var_model = ValueAtRisk(returns, weights)

    hist = var_model.historical_var()
    param = var_model.parametric_var()
    mc = var_model.monte_carlo_var()

    assert approx_equal(hist, param, tol=0.003)
    assert approx_equal(param, mc, tol=0.003)


def test_var_increases_with_confidence_level():
    returns = _make_synthetic_returns()
    weights = np.array([0.4, 0.3, 0.3])
    var_model = ValueAtRisk(returns, weights)

    var_95 = var_model.historical_var(confidence=0.95)
    var_99 = var_model.historical_var(confidence=0.99)
    assert var_99 > var_95, "99% VaR should be larger (more extreme) than 95% VaR"


def test_var_scales_with_sqrt_time():
    returns = _make_synthetic_returns()
    weights = np.array([0.4, 0.3, 0.3])
    var_model = ValueAtRisk(returns, weights)

    var_1day = var_model.parametric_var(horizon_days=1)
    var_10day = var_model.parametric_var(horizon_days=10)
    assert approx_equal(var_10day, var_1day * np.sqrt(10), tol=1e-6)


def test_expected_shortfall_exceeds_var():
    # ES must be >= VaR by definition -- it's the average of losses BEYOND
    # the VaR threshold, so it can never be smaller
    returns = _make_synthetic_returns()
    weights = np.array([0.4, 0.3, 0.3])
    var_model = ValueAtRisk(returns, weights)

    var_95 = var_model.historical_var(confidence=0.95)
    es_95 = var_model.expected_shortfall(confidence=0.95)
    assert es_95 >= var_95


def test_single_asset_portfolio_matches_asset_volatility():
    # With 100% weight in one asset, portfolio VaR should just reflect
    # that asset's own volatility
    returns = _make_synthetic_returns(n_assets=1)
    weights = np.array([1.0])
    var_model = ValueAtRisk(returns, weights)

    from scipy.stats import norm
    expected_param_var = -norm.ppf(0.05) * returns.iloc[:, 0].std()
    assert approx_equal(var_model.parametric_var(), expected_param_var, tol=1e-6)


if __name__ == "__main__":
    tests = [
        test_weights_must_sum_to_one,
        test_weights_length_must_match_assets,
        test_var_is_positive_for_normal_returns,
        test_all_three_methods_agree_on_normal_data,
        test_var_increases_with_confidence_level,
        test_var_scales_with_sqrt_time,
        test_expected_shortfall_exceeds_var,
        test_single_asset_portfolio_matches_asset_volatility,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
