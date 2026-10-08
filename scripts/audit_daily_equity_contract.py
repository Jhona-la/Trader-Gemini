"""Bounded behavioral witnesses for the R4 planning wave.

Compile and call frozen historical Rust selection_stats Git blobs (not a Python port).
Daily arithmetic is extracted exactly from that committed harness and compiled
against a controlled cash/unrealized fixture. This does not validate current
changed code, replay the engine, prove trade incidence or estimate performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ref", default="8938cf41d5b2d181e807de1a3fa41f36b09dfc32",
                        help="Historical committed source to reproduce, not a current-code verdict")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    snapshot = command("git", "-C", str(repo), "rev-parse", "--verify", args.ref + "^{commit}").strip()
    selection_path = "crates/risk-engine/src/selection_stats.rs"
    harness_path = "crates/backtest-engine/src/bin/continuous_evolution_backtest.rs"
    source_by_path = {path: subprocess.run(["git", "-C", str(repo), "show", snapshot + ":" + path],
                                           capture_output=True, check=True).stdout
                      for path in (selection_path, harness_path)}
    source = source_by_path[harness_path].decode("utf-8")
    expression = re.search(r"let day_pnl = ([^;]+);", source)
    if not expression or expression.group(1).strip() != "total_equity - day_start_cap_real":
        raise SystemExit("Committed harness expression changed: re-audit before reusing this witness")
    module_path = "selection_stats.rs"
    rust = r'''
#[path = "MODULE_PATH"]
mod selection_stats;
fn main() {
    let returns: Vec<f64> = (0..400).map(|i| if i % 2 == 0 {0.0026} else {-0.0014}).collect();
    let clean = selection_stats::edge_survives_multiplicity(&returns, 100);
    let mut contaminated = returns.clone();
    contaminated.extend([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    let dirty = selection_stats::edge_survives_multiplicity(&contaminated, 100);
    let short = vec![0.0026, -0.0014].repeat(20);
    let repeated: Vec<f64> = short.iter().flat_map(|x| std::iter::repeat_n(*x, 10)).collect();
    let a = selection_stats::edge_survives_multiplicity(&short, 100);
    let b = selection_stats::edge_survives_multiplicity(&repeated, 100);
    let mut daily_pnl_sum = 0.0;
    for floating in [10.0, 10.0] {
        let day_start_cap_real = 100.0;
        let total_equity = 100.0 + floating;
        let day_pnl = DAILY_EXPRESSION;
        daily_pnl_sum += day_pnl;
    }
    let terminal_growth = 110.0 - 100.0;
    println!("{{\"clean_dsr\":{},\"contaminated_dsr\":{},\"clean_passes\":{},\"contaminated_passes\":{},\"original_observations\":{},\"repeated_observations\":{},\"original_dsr\":{},\"repeated_dsr\":{},\"original_passes\":{},\"repeated_passes\":{},\"daily_pnl_sum\":{},\"terminal_growth\":{}}}",
        clean.dsr, dirty.dsr, clean.passes, dirty.passes,
        short.len(), repeated.len(), a.dsr, b.dsr, a.passes, b.passes,
        daily_pnl_sum, terminal_growth);
}
'''.replace("MODULE_PATH", module_path).replace("DAILY_EXPRESSION", expression.group(1))
    # All files here belong to an ephemeral diagnostic fixture, never a personal directory.
    with tempfile.TemporaryDirectory(prefix="sol-equity-contract-", dir=repo) as temp:
        directory = Path(temp)
        program = directory / "witness.rs"
        executable = directory / "witness.exe"
        program.write_text(rust, encoding="utf-8")
        (directory / "selection_stats.rs").write_bytes(source_by_path[selection_path])
        command("rustc", "--edition=2024", str(program), "-o", str(executable))
        observed = json.loads(command(str(executable)))
    checks = {
        "nonfinite_contamination_not_rejected": observed["contaminated_passes"] and observed["clean_dsr"] == observed["contaminated_dsr"],
        "repeated_observations_create_gate_pass": not observed["original_passes"] and observed["repeated_passes"],
        "carried_unrealized_breaks_daily_reconciliation": observed["daily_pnl_sum"] != observed["terminal_growth"],
    }
    result = {
        "inspected_commit": snapshot,
        "execution_head": command("git", "-C", str(repo), "rev-parse", "HEAD").strip(),
        "rustc": command("rustc", "--version").strip(),
        "scope": "Actual Rust selection_stats from frozen historical Git blobs; exact daily expression in controlled fixture. NOT a current-code verdict, trading-engine replay or profitability.",
        "source_sha256": {path: hashlib.sha256(content).hexdigest() for path, content in source_by_path.items()},
        "source_expression": expression.group(0),
        "observed": observed,
        "witnesses_reproduced": checks,
        "all_witnesses_reproduced": all(checks.values()),
        "economic_validation": "not_executed",
    }
    output = Path(args.output)
    if not output.parent.is_dir():
        raise SystemExit("Output parent must already exist")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
