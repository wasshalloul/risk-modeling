"""
Unit tests for the stress testing module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from risk.stress_testing import StressTester, CRISIS_SHOCKS


def approx_equal(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_weights_must_sum_to_one():
    try:
        StressTester({"AAPL": 0.5, "JPM": 0.6})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_unknown_scenario_raises():
    tester = StressTester({"AAPL": 1.0})
    try:
        tester.apply_scenario("made_up_crisis")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_missing_ticker_shock_data_raises():
    tester = StressTester({"FAKETICKER": 1.0})
    try:
        tester.apply_scenario("2008_financial_crisis")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_full_allocation_to_one_asset_matches_its_shock():
    # 100% in AAPL should just return AAPL's own crisis shock
    tester = StressTester({"AAPL": 1.0}, portfolio_value=1_000_000)
    result = tester.apply_scenario("2008_financial_crisis")
    expected_pct = CRISIS_SHOCKS["2008_financial_crisis"]["AAPL"] * 100
    assert approx_equal(result["portfolio_return_pct"], expected_pct, tol=1e-6)


def test_dollar_pnl_matches_percent_return():
    tester = StressTester({"JPM": 1.0}, portfolio_value=500_000)
    result = tester.apply_scenario("2020_covid_crash")
    expected_dollar = CRISIS_SHOCKS["2020_covid_crash"]["JPM"] * 500_000
    assert approx_equal(result["dollar_pnl"], round(expected_dollar, 2), tol=1.0)


def test_correlation_breakdown_increases_var():
    # Higher stressed correlation should always produce >= VaR vs normal times,
    # since assets are (on average) less diversifying against each other
    weights = {"AAPL": 0.5, "JPM": 0.5}
    tester = StressTester(weights)

    rng = np.random.default_rng(2)
    corr = np.array([[1.0, 0.2], [0.2, 1.0]])
    vols = np.array([0.02, 0.02])
    cov = np.outer(vols, vols) * corr
    returns = pd.DataFrame(
        rng.multivariate_normal([0, 0], cov, size=1000), columns=["AAPL", "JPM"]
    )

    result = tester.correlation_breakdown_var(returns, stressed_corr=0.95)
    assert result["stressed_var_pct"] > result["normal_var_pct"]


def test_perfect_correlation_equals_weighted_vol_sum():
    # At correlation = 1, portfolio vol = weighted sum of individual vols
    # (no diversification benefit at all)
    weights = {"A": 0.6, "B": 0.4}
    tester = StressTester(weights)

    rng = np.random.default_rng(3)
    vols = np.array([0.02, 0.03])
    corr = np.array([[1.0, 0.1], [0.1, 1.0]])
    cov = np.outer(vols, vols) * corr
    returns = pd.DataFrame(
        rng.multivariate_normal([0, 0], cov, size=1000), columns=["A", "B"]
    )

    result = tester.correlation_breakdown_var(returns, stressed_corr=1.0)
    # weighted vol sum (as a %, 1-day, 95% confidence) -- using the SAMPLE
    # vols actually computed inside the method, not the true generating
    # vols, since finite-sample noise makes those differ slightly
    from scipy.stats import norm
    sample_vols = returns[["A", "B"]].std().values
    expected_std = weights["A"] * sample_vols[0] + weights["B"] * sample_vols[1]
    expected_var_pct = -norm.ppf(0.05) * expected_std * 100
    assert approx_equal(result["stressed_var_pct"], round(expected_var_pct, 3), tol=0.01)


if __name__ == "__main__":
    tests = [
        test_weights_must_sum_to_one,
        test_unknown_scenario_raises,
        test_missing_ticker_shock_data_raises,
        test_full_allocation_to_one_asset_matches_its_shock,
        test_dollar_pnl_matches_percent_return,
        test_correlation_breakdown_increases_var,
        test_perfect_correlation_equals_weighted_vol_sum,
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
