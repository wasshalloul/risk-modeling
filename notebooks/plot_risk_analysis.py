"""
Visualize VaR methodology comparison, the portfolio return distribution
with VaR/ES marked, and the stress test scenario results.

Usage:
    python plot_risk_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from risk.var_models import ValueAtRisk

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

returns = pd.read_csv(os.path.join(data_dir, "portfolio_returns.csv"), index_col=0)
with open(os.path.join(data_dir, "analysis_summary.json")) as f:
    summary = json.load(f)

weights_dict = summary["weights"]
tickers = list(weights_dict.keys())
weights = np.array([weights_dict[t] for t in tickers])

var_model = ValueAtRisk(returns[tickers], weights)
portfolio_returns = var_model.portfolio_returns * 100  # in %

confidence = summary["confidence"]
var_summary = summary["var_summary"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Panel 1: Return distribution with VaR/ES marked ---
axes[0, 0].hist(portfolio_returns, bins=60, color="#93c5fd", edgecolor="white", alpha=0.8)
axes[0, 0].axvline(-var_summary["historical_var_pct"], color="#dc2626", linestyle="--",
                    label=f"Historical VaR ({int(confidence*100)}%)")
axes[0, 0].axvline(-var_summary["expected_shortfall_pct"], color="#7c2d12", linestyle="--",
                    label="Expected Shortfall")
axes[0, 0].set_title("Portfolio Daily Return Distribution")
axes[0, 0].set_xlabel("Daily return (%)")
axes[0, 0].legend(fontsize=8)

# --- Panel 2: VaR method comparison ---
methods = ["historical_var_pct", "parametric_var_pct", "monte_carlo_var_pct"]
labels = ["Historical", "Parametric", "Monte Carlo"]
values = [var_summary[m] for m in methods]
axes[0, 1].bar(labels, values, color=["#2563eb", "#16a34a", "#9333ea"])
axes[0, 1].set_title(f"1-Day VaR by Method ({int(confidence*100)}% confidence)")
axes[0, 1].set_ylabel("VaR (%)")
for i, v in enumerate(values):
    axes[0, 1].text(i, v + 0.02, f"{v:.2f}%", ha="center", fontsize=9)

# --- Panel 3: Crisis scenario P&L ---
scenarios = summary["scenarios"]
scenario_names = list(scenarios.keys())
scenario_returns = [scenarios[s]["portfolio_return_pct"] for s in scenario_names]
axes[1, 0].barh(scenario_names, scenario_returns, color="#dc2626")
axes[1, 0].set_title("Historical Crisis Scenario: Portfolio Return")
axes[1, 0].set_xlabel("Portfolio return (%)")
for i, v in enumerate(scenario_returns):
    axes[1, 0].text(v, i, f" {v}%", va="center", fontsize=9)

# --- Panel 4: Correlation breakdown ---
corr_stress = summary["correlation_stress"]
corr_labels = ["Normal\ncorrelation", "Stressed\ncorrelation"]
corr_values = [corr_stress["normal_var_pct"], corr_stress["stressed_var_pct"]]
axes[1, 1].bar(corr_labels, corr_values, color=["#16a34a", "#dc2626"])
axes[1, 1].set_title(f"VaR Under Correlation Breakdown\n(+{corr_stress['var_increase_pct']}% when correlations spike)")
axes[1, 1].set_ylabel("VaR (%)")
for i, v in enumerate(corr_values):
    axes[1, 1].text(i, v + 0.02, f"{v:.2f}%", ha="center", fontsize=9)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "risk_analysis.png")
plt.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")
