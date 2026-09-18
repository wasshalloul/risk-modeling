"""
Stress Testing
------------------
VaR tells you about "normal" bad days (the 5th percentile of a typical
distribution). Stress testing asks a different question: what happens to
this portfolio in a SPECIFIC historical crisis scenario? This matters
because real crises often move markets in ways VaR models -- calibrated on
calmer historical data -- don't fully capture (correlations spike, "safe"
diversification breaks down).

We apply two kinds of stress:
1. Historical scenario shocks: apply the actual % move each asset class
   experienced during a real crisis window (e.g. Sept-Nov 2008, or the
   Feb-Mar 2020 COVID crash) to the current portfolio weights.
2. Correlation breakdown: show how portfolio risk changes if correlations
   between assets increase toward 1 during a crisis (a common real
   phenomenon -- diversification tends to disappear exactly when you need
   it most).
"""

import numpy as np
import pandas as pd


# Approximate historical peak-to-trough returns for these tickers during
# two well-known crisis windows. These are illustrative approximations for
# educational stress testing, not exact tick-by-tick historical data.
CRISIS_SHOCKS = {
    "2008_financial_crisis": {
        # Sept 2008 - Mar 2009 approx peak-to-trough
        "AAPL": -0.55, "JPM": -0.75, "XOM": -0.40, "JNJ": -0.15, "WMT": -0.10,
    },
    "2020_covid_crash": {
        # Feb 19 2020 - Mar 23 2020
        "AAPL": -0.31, "JPM": -0.42, "XOM": -0.60, "JNJ": -0.24, "WMT": -0.13,
    },
}


class StressTester:
    def __init__(self, weights: dict, portfolio_value=1_000_000):
        """
        Parameters
        ----------
        weights : dict   {ticker: weight}, weights must sum to 1
        portfolio_value : float   current portfolio value in dollars
        """
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("weights must sum to 1")
        self.weights = weights
        self.portfolio_value = portfolio_value

    def apply_scenario(self, scenario_name: str):
        """
        Apply a named historical crisis scenario to the current portfolio
        weights and return the estimated portfolio loss.
        """
        if scenario_name not in CRISIS_SHOCKS:
            raise ValueError(f"Unknown scenario '{scenario_name}'. "
                              f"Available: {list(CRISIS_SHOCKS.keys())}")

        shocks = CRISIS_SHOCKS[scenario_name]
        missing = set(self.weights) - set(shocks)
        if missing:
            raise ValueError(f"No shock data for tickers: {missing}")

        portfolio_return = sum(self.weights[t] * shocks[t] for t in self.weights)
        dollar_loss = portfolio_return * self.portfolio_value

        per_asset = {
            t: {"weight": self.weights[t], "shock_pct": shocks[t] * 100,
                "contribution_pct": self.weights[t] * shocks[t] * 100}
            for t in self.weights
        }

        return {
            "scenario": scenario_name,
            "portfolio_return_pct": round(portfolio_return * 100, 2),
            "dollar_pnl": round(dollar_loss, 2),
            "per_asset_breakdown": per_asset,
        }

    def correlation_breakdown_var(self, returns: pd.DataFrame, stressed_corr=0.9,
                                    confidence=0.95):
        """
        Recompute parametric VaR assuming ALL pairwise correlations jump to
        `stressed_corr` (simulating a crisis where diversification breaks
        down), keeping each asset's own volatility unchanged. Compares this
        to normal-times VaR to show the added risk from correlation spikes.
        """
        from scipy.stats import norm

        tickers = list(self.weights.keys())
        w = np.array([self.weights[t] for t in tickers])
        vols = returns[tickers].std().values

        # Normal-times covariance (from actual historical correlation)
        normal_cov = returns[tickers].cov().values
        normal_var = w @ normal_cov @ w
        normal_std = np.sqrt(normal_var)

        # Stressed covariance: same vols, but all correlations forced to
        # `stressed_corr` (except diagonal, which stays 1)
        stressed_corr_matrix = np.full((len(tickers), len(tickers)), stressed_corr)
        np.fill_diagonal(stressed_corr_matrix, 1.0)
        stressed_cov = np.outer(vols, vols) * stressed_corr_matrix
        stressed_var = w @ stressed_cov @ w
        stressed_std = np.sqrt(stressed_var)

        z = norm.ppf(1 - confidence)
        normal_var_pct = -z * normal_std
        stressed_var_pct = -z * stressed_std

        return {
            "normal_var_pct": round(normal_var_pct * 100, 3),
            "stressed_var_pct": round(stressed_var_pct * 100, 3),
            "normal_var_dollar": round(normal_var_pct * self.portfolio_value, 2),
            "stressed_var_dollar": round(stressed_var_pct * self.portfolio_value, 2),
            "var_increase_pct": round((stressed_var_pct / normal_var_pct - 1) * 100, 1),
        }


if __name__ == "__main__":
    weights = {"AAPL": 0.25, "JPM": 0.25, "XOM": 0.2, "JNJ": 0.15, "WMT": 0.15}
    tester = StressTester(weights, portfolio_value=1_000_000)

    for scenario in CRISIS_SHOCKS:
        result = tester.apply_scenario(scenario)
        print(f"\n{scenario}:")
        print(f"  Portfolio return: {result['portfolio_return_pct']}%")
        print(f"  Dollar P&L: ${result['dollar_pnl']:,.2f}")

    # Correlation breakdown demo with synthetic data
    rng = np.random.default_rng(0)
    n_assets = 5
    corr = np.full((n_assets, n_assets), 0.25)
    np.fill_diagonal(corr, 1.0)
    vols = np.array([0.02, 0.022, 0.025, 0.012, 0.011])
    cov = np.outer(vols, vols) * corr
    returns = pd.DataFrame(
        rng.multivariate_normal([0.0003] * n_assets, cov, size=1000),
        columns=list(weights.keys())
    )

    corr_result = tester.correlation_breakdown_var(returns, stressed_corr=0.9)
    print("\nCorrelation breakdown stress test:")
    for k, v in corr_result.items():
        print(f"  {k}: {v}")
