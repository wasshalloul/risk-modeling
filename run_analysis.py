"""
Run the full VaR + stress testing analysis on a real portfolio.

Usage:
    python run_analysis.py --tickers AAPL JPM XOM JNJ WMT --weights 0.25 0.25 0.2 0.15 0.15 --years 5
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yfinance as yf
from risk.var_models import ValueAtRisk
from risk.stress_testing import StressTester, CRISIS_SHOCKS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "JPM", "XOM", "JNJ", "WMT"])
    parser.add_argument("--weights", nargs="+", type=float, default=[0.25, 0.25, 0.2, 0.15, 0.15])
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--portfolio-value", type=float, default=1_000_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    if len(args.tickers) != len(args.weights):
        raise ValueError("Number of tickers must match number of weights")
    if not np.isclose(sum(args.weights), 1.0):
        raise ValueError(f"Weights must sum to 1, got {sum(args.weights)}")

    weights_dict = dict(zip(args.tickers, args.weights))
    weights_array = np.array(args.weights)

    print(f"Downloading {args.years}y of data for {args.tickers}...")
    data = yf.download(args.tickers, period=f"{args.years}y", progress=False)["Close"]
    data = data[args.tickers]  # preserve order
    returns = data.pct_change().dropna()

    print(f"\n{'='*60}")
    print(f"PORTFOLIO: {weights_dict}")
    print(f"Portfolio value: ${args.portfolio_value:,.0f}")
    print(f"{'='*60}")

    # --- VaR ---
    var_model = ValueAtRisk(returns, weights_array)
    var_summary = var_model.summary(confidence=args.confidence, portfolio_value=args.portfolio_value)

    print(f"\n--- Value-at-Risk ({int(args.confidence*100)}% confidence, 1-day) ---")
    for k, v in var_summary.items():
        print(f"  {k}: {v}")

    # --- Stress testing ---
    tester = StressTester(weights_dict, portfolio_value=args.portfolio_value)

    print(f"\n--- Historical Crisis Scenarios ---")
    for scenario in CRISIS_SHOCKS:
        result = tester.apply_scenario(scenario)
        print(f"\n{scenario}:")
        print(f"  Portfolio return: {result['portfolio_return_pct']}%")
        print(f"  Dollar P&L: ${result['dollar_pnl']:,.2f}")

    print(f"\n--- Correlation Breakdown Stress Test ---")
    corr_result = tester.correlation_breakdown_var(returns, stressed_corr=0.9,
                                                      confidence=args.confidence)
    for k, v in corr_result.items():
        print(f"  {k}: {v}")

    # Save returns data for plotting
    out_path = os.path.join(os.path.dirname(__file__), "data", "portfolio_returns.csv")
    returns.to_csv(out_path)
    print(f"\nSaved returns data to {out_path}")

    # Save summary for the plotting script
    import json
    summary_out = {
        "weights": weights_dict,
        "portfolio_value": args.portfolio_value,
        "confidence": args.confidence,
        "var_summary": var_summary,
        "scenarios": {s: tester.apply_scenario(s) for s in CRISIS_SHOCKS},
        "correlation_stress": corr_result,
    }
    summary_path = os.path.join(os.path.dirname(__file__), "data", "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Saved analysis summary to {summary_path}")
