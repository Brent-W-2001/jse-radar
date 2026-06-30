"""
One-off script: inspect the fitted Markov chain against real regime
history, and compare expected_regime_duration() against the empirical
average spell lengths already computed in 02_regime_analysis.ipynb.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jse_radar.modeling.markov_chain import RegimeMarkovChain

chain = RegimeMarkovChain().fit()

print(f"{'Regime':30s} | {'Occurrences':>11s} | {'Reliable':>8s} | {'Expected Duration':>18s}")
print("-" * 80)

for regime in chain.regimes:
    conf = chain.confidence_for(regime)
    duration = chain.expected_regime_duration(regime)
    duration_str = f"{duration:.1f} months" if duration == duration else "N/A"
    if duration == float("inf"):
        duration_str = "infinite"

    print(
        f"{regime:30s} | {conf['historical_occurrences']:>11d} | "
        f"{str(conf['reliable']):>8s} | {duration_str:>18s}"
    )