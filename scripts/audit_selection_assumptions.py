#!/usr/bin/env python3
"""Reproduce R4 selection-statistics counterexamples with Python's stdlib.

This is a mathematical port and a trace-level model, NOT execution of the
production Rust, its trading engine, or an out-of-sample performance experiment.
NormalDist supplies normal CDF/quantiles instead of Rust's approximations.
No third-party packages, network requests, builds, or live trading are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess


NORMAL = statistics.NormalDist()
EULER_GAMMA = 0.5772156649015329
BASE_COMMIT = "adeb8d1b1f8171b14fffa8abc9f54c396e4794dd"
SOURCE_FILES = (
    "crates/risk-engine/src/selection_stats.rs",
    "crates/god-engine-core/src/darwin.rs",
    "crates/god-engine-core/src/lib.rs",
    "crates/evolution-engine/src/online_daemon.rs",
    "src/bin/god_engine.rs",
)


def dsr_formula_port(returns: list[float], n_trials: int) -> float:
    """Port of the current formula; makes NO independence-validity claim."""
    clean = [value for value in returns if math.isfinite(value)]
    if len(clean) < 20:
        return 0.0
    n = len(clean)
    mean = sum(clean) / n
    sd = math.sqrt(sum((value - mean) ** 2 for value in clean) / (n - 1))
    if sd <= 1e-12:
        return 0.0
    sr = mean / sd
    skewness = sum(((value - mean) / sd) ** 3 for value in clean) / n
    kurtosis = sum(((value - mean) / sd) ** 4 for value in clean) / n
    denom_sq = 1 - skewness * sr + (kurtosis - 1) * sr * sr / 4
    se = math.sqrt(denom_sq / (n - 1)) if denom_sq > 1e-12 else 1 / math.sqrt(n - 1)
    trials = max(n_trials, 2)
    expected_max_z = (
        (1 - EULER_GAMMA) * NORMAL.inv_cdf(1 - 1 / trials)
        + EULER_GAMMA * NORMAL.inv_cdf(1 - 1 / (trials * math.e))
    )
    return NORMAL.cdf(sr / se - expected_max_z)


def source_provenance(repo: Path) -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    return {
        "review_base_commit": BASE_COMMIT,
        "execution_checkout_head": head,
        "source_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES
        },
    }


def duplication_probe() -> dict:
    original = [0.0026, -0.0014] * 20
    repeated = [value for value in original for _ in range(10)]
    return {
        "n_trials": 100,
        "original_observations": len(original),
        "original_dsr": dsr_formula_port(original, 100),
        "repeated_observations": len(repeated),
        "repeat_factor": 10,
        "repeated_dsr": dsr_formula_port(repeated, 100),
        "interpretation": "Repeated values add no independent innovations; nominal confidence increases.",
    }


def ar1_null_probe(seed: int, simulations: int, observations: int) -> list[dict]:
    # One generator and fixed phi order reproduce the published audit numbers.
    # x0 ~ N(0,1) and innovation sd sqrt(1-phi^2) give stationary marginal N(0,1).
    rng = random.Random(seed)
    results = []
    for phi in (0.0, 0.9, 0.95):
        scores = []
        for _ in range(simulations):
            x = rng.gauss(0, 1)
            series = []
            for _ in range(observations):
                x = phi * x + math.sqrt(1 - phi * phi) * rng.gauss(0, 1)
                series.append(x * 0.001)
            scores.append(dsr_formula_port(series, 100))
        passing = sum(score >= 0.95 for score in scores)
        results.append({
            "phi": phi,
            "population_mean": 0.0,
            "observations": observations,
            "simulations": simulations,
            "assumed_n_trials": 100,
            "passing_dsr_ge_0_95": passing,
            "passing_fraction": passing / simulations,
            "median_dsr": statistics.median(scores),
        })
    return results


def realized_clock_trace() -> dict:
    # Exogenous account trace; not a claim that engine orders generate this path.
    trace = [
        (0, 100.0, 0.0, False),
        (500, 100.0, -20.0, False),
        (1000, 100.0, -40.0, False),
        (1500, 100.0, -10.0, False),
        (2000, 100.0, 0.0, False),
        (2100, 101.0, 0.0, True),
        (2200, 100.0, 0.0, True),
        (3000, 100.0, 0.0, False),
        (3200, 100.0, 0.0, False),
    ]
    last_sample_ts = trace[0][0]
    prev_capital = 100.0
    cash_peak = equity_peak = 100.0
    cash_dd = equity_dd = 0.0
    samples = []
    for ts, cash, unrealized, closed in trace:
        equity = cash + unrealized
        equity_peak = max(equity_peak, equity)
        equity_dd = max(equity_dd, (equity_peak - equity) / equity_peak)
        if closed:
            cash_peak = max(cash_peak, cash)
            cash_dd = max(cash_dd, (cash_peak - cash) / cash_peak)
        if ts >= last_sample_ts + 1000 or closed:
            samples.append({
                "timestamp_ms": ts,
                "delta_t_ms": ts - last_sample_ts,
                "realized_return": (cash - prev_capital) / prev_capital,
                "marked_equity": equity,
            })
            prev_capital = cash
            last_sample_ts = ts
    return {
        "scope": "Port of darwin.rs sampling/drawdown expressions on an exogenous account trace.",
        "input": [dict(timestamp_ms=t, cash=c, unrealized=u, closed=k) for t, c, u, k in trace],
        "samples": samples,
        "reported_realized_drawdown": cash_dd,
        "marked_equity_drawdown": equity_dd,
    }


def reset_counter_trace() -> dict:
    # Mirrors host lifecycle: new() inside each loop, each batch = 20*5 trials.
    seen = []
    for _ in range(3):
        cumulative_trials = 0
        cumulative_trials += 20 * 5
        seen.append(cumulative_trials)
    return {
        "scope": "Lifecycle trace derived from src/bin/god_engine.rs:1960-1965; not Rust execution.",
        "host_round_counts": seen,
        "persistent_instance_expected_counts": [100, 200, 300],
        "execution_requires": "ENABLE_LEGACY_DARWIN_DAEMON=true",
        "publication_also_requires": "ENABLE_ONLINE_DARWIN_MUTATION=true or 1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--observations", type=int, default=500)
    args = parser.parse_args()
    if args.simulations <= 0 or args.observations < 20:
        parser.error("simulations must be positive and observations must be >=20")
    result = {
        "audit": "R4 selection assumptions, 2026-10-07",
        "evidence_kind": "Python mathematical port + trace model; not Rust execution or trading results",
        "cdf_implementation": "Python statistics.NormalDist, not Rust Acklam/Abramowitz-Stegun approximations",
        "provenance": source_provenance(args.repo.resolve()),
        "seed": args.seed,
        "duplication": duplication_probe(),
        "ar1_null": ar1_null_probe(args.seed, args.simulations, args.observations),
        "realized_clock": realized_clock_trace(),
        "counter_lifecycle": reset_counter_trace(),
        "target_units": {
            "log_growth_per_day": math.log(2) / 3,
            "equivalent_daily_simple_return": 2 ** (1 / 3) - 1,
            "idealized_gbm_full_kelly_daily_signal_to_noise": math.sqrt(2 * math.log(2) / 3),
            "assumptions": "Continuous-time GBM, known constant drift/volatility, zero costs/risk-free rate, unrestricted full Kelly; not a guarantee or an observed Sharpe.",
        },
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
