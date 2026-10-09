#!/usr/bin/env python3
"""Reproduce R4 diagnostics on immutable Rust blobs, without Cargo or live APIs.

These assertions characterize known defects in 18bbd1d9. A successful run
means the diagnostics reproduced, not that production is repaired or profitable.
Only a new, empty output directory is written. Compilers are invoked via rustup.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

SOURCE_REF = "18bbd1d90a073ae49ca8083c5898537013e4bfeb"
TOOLCHAIN = "nightly-2026-06-30"
SOURCES = {
    "selection_stats.rs": ("crates/risk-engine/src/selection_stats.rs", "1e2ba9b194cf133bc614071c168003287558145d"),
    "types.rs": ("crates/strategy-core/src/types.rs", "5ca8505c383039f06f57090f86c21b3e2a1d10a5"),
    "multivariate_coint.rs": ("crates/strategy-core/src/multivariate_coint.rs", "3d87089163cf2011e7e9a4efb7d98c88985d4a51"),
    "vecm_arbitrage.rs": ("crates/strategy-core/src/vecm_arbitrage.rs", "f3a2cc9d344995b7cdf40f7d96324fe2b42a3f4a"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output-dir", help="Must be absent or empty; default is a unique directory under target")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else repo / "target" / "r4-rust-witnesses" / str(uuid.uuid4())
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty output directory")
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "source_commit": SOURCE_REF,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "scope": "Immutable-source isolated Rust diagnostics; no Cargo, live pipeline, market replay or economic result",
        "status": "IN_PROGRESS", "sources": {}, "commands": [],
    }
    receipt_path = output / "receipt.json"

    def save() -> None:
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def run(label: str, argv: list[str]) -> None:
        start = time.monotonic()
        result = subprocess.run(argv, cwd=output, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        log_path = output / f"{label}.log"
        log_path.write_bytes(result.stdout)
        receipt["commands"].append({"label": label, "argv": argv, "exit_code": result.returncode,
                                    "elapsed_seconds": round(time.monotonic() - start, 6),
                                    "log": log_path.name, "log_sha256": sha(result.stdout)})
        save()
        if result.returncode:
            raise RuntimeError(f"{label} failed with exit {result.returncode}; see {log_path}")

    try:
        for name, (path, expected_blob) in SOURCES.items():
            blob = git(repo, "rev-parse", f"{SOURCE_REF}:{path}").decode().strip()
            if blob != expected_blob:
                raise ValueError(f"Unexpected immutable blob for {path}")
            whole = git(repo, "show", f"{SOURCE_REF}:{path}")
            copied = whole
            bounds = None
            if name == "vecm_arbitrage.rs":
                # Exact SDE implementation only; unrelated VECM imports require the workspace.
                bounds = [197, 337]
                copied = b"".join(whole.splitlines(keepends=True)[196:337])
                if sha(copied) != "dabb8d2a2c86df2ec57fd7b70420893399538056924c953b5e892393619e49e3":
                    raise ValueError("SDE extraction no longer matches the reviewed fragment")
            (output / name).write_bytes(copied)
            receipt["sources"][name] = {"path": path, "blob": blob, "whole_source_sha256": sha(whole),
                                        "copied_sha256": sha(copied), "extracted_lines": bounds}
        fixtures = Path(__file__).resolve().parent / "fixtures"
        for name in ("r4_selection_stats_witness.rs", "r4_ou_witness.rs"):
            raw = (fixtures / name).read_bytes()
            (output / name).write_bytes(raw)
            receipt.setdefault("wrappers", {})[name] = sha(raw)
        compiler = ["rustup", "run", TOOLCHAIN, "rustc"]
        run("00-toolchain", compiler + ["--version", "--verbose"])
        suffix = ".exe" if os.name == "nt" else ""
        selection_exe = output / ("selection_witness" + suffix)
        ou_exe = output / ("ou_witness" + suffix)
        run("01-selection-compile", compiler + ["--edition=2021", "r4_selection_stats_witness.rs", "-o", str(selection_exe)])
        run("02-selection-run", [str(selection_exe)])
        run("03-ou-compile", compiler + ["--edition=2021", "--test", "r4_ou_witness.rs", "-o", str(ou_exe)])
        run("04-ou-run", [str(ou_exe), "r4_ou_diagnostic::", "--nocapture", "--test-threads=1"])
        receipt["status"] = "DIAGNOSTICS_REPRODUCED_NOT_REPAIRED"
        save()
        print(json.dumps({"receipt": str(receipt_path), "status": receipt["status"]}))
        return 0
    except Exception as error:
        receipt["status"] = "INCOMPLETE"
        receipt["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
