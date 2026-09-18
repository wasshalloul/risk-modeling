"""
Value-at-Risk (VaR) Models
------------------------------
Three standard approaches to estimating portfolio VaR, each with different
assumptions and trade-offs:

1. Historical Simulation
   Uses actual historical portfolio returns directly, with no distributional
   assumption. Simple and captures real fat tails / skew in the data, but
   entirely dependent on the historical window used -- it assumes the past
   is representative of the future.

2. Parametric (Variance-Covariance)
   Assumes portfolio returns are normally distributed, computes portfolio
   volatility from the asset covariance matrix, and uses the normal
   distribution's z-score to get VaR analytically. Fast and simple, but
   underestimates risk when real returns have fat tails (which they almost
   always do).

3. Monte Carlo Simulation
   Simulates many possible future return paths using the estimated mean and
   covariance structure (assuming multivariate normal returns here), then
   reads VaR off the simulated distribution. More flexible than parametric
   (can incorporate more complex return models), but computationally heavier
   and still only as good as the assumed return-generating process.

VaR interpretation: "1-day 95% VaR of $X" means we expect to lose no more
than $X on 95% of days -- equivalently, a loss exceeding $X is expected on
about 5% of days (1 in 20).
"""

import numpy as np
import pandas as pd


class ValueAtRisk:
    def __init__(self, returns: pd.DataFrame, weights: np.ndarray):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns for each asset, one column per asset.
        weights : np.ndarray
            Portfolio weights, must sum to 1, same order as returns.columns.
        """
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("weights must sum to 1")
        if len(weights) != returns.shape[1]:
            raise ValueError("weights length must match number of assets")

        self.returns = returns
        self.weights = weights

        # Portfolio daily returns: weighted sum of asset returns each day
        self.portfolio_returns = returns.values @ weights

    def historical_var(self, confidence=0.95, horizon_days=1):
        """
        Historical simulation VaR: directly take the empirical percentile
        of past portfolio returns, scaled to the desired horizon by sqrt(time).
        """
        var_pct = -np.percentile(self.portfolio_returns, (1 - confidence) * 100)
        return var_pct * np.sqrt(horizon_days)

    def parametric_var(self, confidence=0.95, horizon_days=1):
        """
        Parametric (variance-covariance) VaR, assuming normally distributed
        portfolio returns.
        """
        from scipy.stats import norm

        cov_matrix = self.returns.cov().values
        portfolio_var = self.weights @ cov_matrix @ self.weights
        portfolio_std = np.sqrt(portfolio_var)

        z_score = norm.ppf(1 - confidence)  # negative number, e.g. -1.645 for 95%
        var_pct = -z_score * portfolio_std
        return var_pct * np.sqrt(horizon_days)

    def monte_carlo_var(self, confidence=0.95, horizon_days=1, n_simulations=50_000, seed=42):
        """
        Monte Carlo VaR: simulate portfolio returns assuming a multivariate
        normal distribution fitted to the historical mean and covariance,
        then take the empirical percentile of the simulated outcomes.
        """
        rng = np.random.default_rng(seed)
        mean_returns = self.returns.mean().values
        cov_matrix = self.returns.cov().values

        simulated_asset_returns = rng.multivariate_normal(
            mean_returns, cov_matrix, size=n_simulations
        )
        simulated_portfolio_returns = simulated_asset_returns @ self.weights

        var_pct = -np.percentile(simulated_portfolio_returns, (1 - confidence) * 100)
        return var_pct * np.sqrt(horizon_days)

    def expected_shortfall(self, confidence=0.95, horizon_days=1, method="historical"):
        """
        Expected Shortfall (a.k.a. Conditional VaR): the AVERAGE loss in the
        worst (1-confidence) fraction of cases, not just the threshold.
        This addresses a key weakness of VaR -- it says nothing about how
        bad losses get beyond the VaR threshold, only that they occur.
        """
        if method == "historical":
            threshold = np.percentile(self.portfolio_returns, (1 - confidence) * 100)
            tail_losses = self.portfolio_returns[self.portfolio_returns <= threshold]
            es_pct = -tail_losses.mean()
        else:
            raise NotImplementedError("Only 'historical' method is implemented for ES")

        return es_pct * np.sqrt(horizon_days)

    def summary(self, confidence=0.95, horizon_days=1, portfolio_value=1_000_000):
        """
        Convenience method: compute all three VaR estimates plus Expected
        Shortfall, both as a percentage and in dollar terms for a given
        portfolio value.
        """
        hist = self.historical_var(confidence, horizon_days)
        param = self.parametric_var(confidence, horizon_days)
        mc = self.monte_carlo_var(confidence, horizon_days)
        es = self.expected_shortfall(confidence, horizon_days)

        return {
            f"historical_var_pct": round(hist * 100, 3),
            f"parametric_var_pct": round(param * 100, 3),
            f"monte_carlo_var_pct": round(mc * 100, 3),
            f"expected_shortfall_pct": round(es * 100, 3),
            f"historical_var_dollar": round(hist * portfolio_value, 2),
            f"parametric_var_dollar": round(param * portfolio_value, 2),
            f"monte_carlo_var_dollar": round(mc * portfolio_value, 2),
            f"expected_shortfall_dollar": round(es * portfolio_value, 2),
        }


if __name__ == "__main__":
    # Sanity check with synthetic correlated returns
    rng = np.random.default_rng(0)
    n_days = 1000
    n_assets = 5

    # Build a synthetic covariance structure: mild positive correlation
    corr = np.full((n_assets, n_assets), 0.3)
    np.fill_diagonal(corr, 1.0)
    vols = np.array([0.015, 0.012, 0.020, 0.010, 0.011])  # daily vols
    cov = np.outer(vols, vols) * corr

    returns = pd.DataFrame(
        rng.multivariate_normal(mean=[0.0004] * n_assets, cov=cov, size=n_days),
        columns=["AAPL", "JPM", "XOM", "JNJ", "WMT"]
    )
    weights = np.array([0.25, 0.25, 0.2, 0.15, 0.15])

    var_model = ValueAtRisk(returns, weights)
    summary = var_model.summary(confidence=0.95, portfolio_value=1_000_000)

    print("VaR Summary (synthetic data, $1M portfolio, 95% confidence, 1-day):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
