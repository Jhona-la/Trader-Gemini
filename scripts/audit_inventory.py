#!/usr/bin/env python3
"""Metadata-only, reproducible audit ledger for a fixed Git commit.

generate starts every file as inventoried, never reviewed. --previous carries
evidence forward only for identical Git objects/modes and preserves invalidated
or removed reviews. check validates coverage and structure, not the truth of a
human review. delta compares committed trees; uncommitted work is out of scope.
No source, credentials, data sets, or logs are copied into the ledger.
"""

import argparse
import copy
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


SCHEMA_VERSION = 1
STATUSES = {"inventariado", "en_revision", "hallazgo_abierto", "verificado"}
MATH_FILES = {
    "genome.rs", "temporal_spectrum.rs", "espectral_multiactivo.rs",
    "evalues.rs", "skill_motores.rs", "cramer_lundberg.rs", "ruin.rs",
    "drawdown.rs", "capital_regime.rs", "calibration.rs", "random_matrix.rs",
    "hodge.rs", "multifractal.rs", "spectral_tape.rs", "selection_stats.rs",
    "veto_registry.rs", "micro_weight.rs",
}
PHASE_BY_CRATE = {
    "quantum-arena": "F2", "feature-engine": "F2", "signal-engine": "F2",
    "strategy-core": "F2", "god-engine-core": "F3", "risk-engine": "F4",
    "execution-engine": "F4", "evolution-engine": "F5",
    "backtest-engine": "F5", "dark-alpha-engine": "F5",
    "data-pipeline": "F6", "data-ingest": "F6", "storage-engine": "F6",
    "metacortex-engine": "F6",
}
OWNER_BY_PHASE = {
    "F0": "consejo_documentacion", "F1": "consejo_matematica_estadistica",
    "F2": "consejo_espectro_motores", "F3": "consejo_integracion",
    "F4": "consejo_riesgo_ejecucion", "F5": "consejo_evolucion_validacion",
    "F6": "consejo_datos_persistencia", "F7": "consejo_plataforma",
    "F8": "consejo_pruebas_independientes",
}


def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def snapshot(repo, ref):
    commit = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    entries = []
    for raw in git(repo, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if not raw:
            continue
        metadata, path = raw.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split()
        entries.append({"path": path.decode("utf-8", errors="surrogateescape"),
                        "git_mode": mode, "git_type": kind, "blob": object_id})
    return commit, sorted(entries, key=lambda row: row["path"])


def classify(path, git_type="blob"):
    parts = PurePosixPath(path).parts
    name = parts[-1]
    suffix = PurePosixPath(path).suffix.lower()
    zone = "/".join(parts[:2]) if parts[0] == "crates" and len(parts) > 1 else parts[0]
    subject_phase = PHASE_BY_CRATE.get(parts[1], "F7") if parts[0] == "crates" else "F7"
    if name in MATH_FILES:
        subject_phase = "F1"
    elif path.startswith("src/bin/god_engine") or path == "src/lib.rs":
        subject_phase = "F3"
    elif parts[0] == "src" and any(word in name for word in ("backtest", "evol", "replay", "train")):
        subject_phase = "F5"
    if git_type == "commit":
        category = "submodule"
    elif parts[0] in {"graphify-out", "artifacts"}:
        category = "generated"
    elif "tests" in parts or name.startswith("test_"):
        category = "test"
    elif suffix == ".md" or parts[0] in {".agents", ".agent"}:
        category = "documentation"
    elif suffix in {".rs", ".py", ".js", ".ts", ".css", ".html"}:
        category = "source"
    elif suffix in {".toml", ".lock", ".yaml", ".yml", ".env", ".conf"} or parts[0] == "config_dir":
        category = "configuration"
    elif suffix in {".ps1", ".bat", ".sh"}:
        category = "operations"
    elif parts[0] in {"data", "models"} or suffix in {".csv", ".parquet", ".gz"}:
        category = "data_or_model"
    else:
        category = "other"
    phase = "F8" if category == "test" else "F0" if category == "documentation" else subject_phase
    risk = "critical" if subject_phase in {"F1", "F3", "F4", "F5", "F6"} else "standard"
    return {"zone": zone, "category": category, "phase": phase,
            "subject_phase": subject_phase, "risk_priority_proposed": risk,
            "owner_proposed": OWNER_BY_PHASE[phase]}


def signature(row):
    return row["blob"], row["git_mode"], row["git_type"]


def load_ledger(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    errors = validate_structure(value)
    if errors:
        raise ValueError("; ".join(errors))
    return value


def validate_structure(ledger):
    errors = []
    if not isinstance(ledger, dict):
        return ["ledger must be an object"]
    if ledger.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if not isinstance(ledger.get("snapshot_commit"), str):
        errors.append("missing snapshot_commit")
    rows = ledger.get("files")
    if not isinstance(rows, list):
        return errors + ["files must be an array"]
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            errors.append("file row must be an object")
            continue
        path = row.get("path")
        if not isinstance(path, str) or not path:
            errors.append("file path must be nonempty text")
            continue
        if path in seen:
            errors.append(f"duplicate path: {path}")
        seen.add(path)
        for key in ("blob", "git_mode", "git_type", "phase", "category", "owner_proposed"):
            if not isinstance(row.get(key), str) or not row.get(key):
                errors.append(f"{path}: missing {key}")
        if row.get("status") not in STATUSES:
            errors.append(f"{path}: invalid status")
        if row.get("phase") not in OWNER_BY_PHASE:
            errors.append(f"{path}: invalid phase")
        for key in ("evidence", "history"):
            if not isinstance(row.get(key), list):
                errors.append(f"{path}: {key} must be an array")
        if row.get("status") == "verificado":
            if not row.get("review_owner") or not row.get("evidence"):
                errors.append(f"{path}: verified status needs review_owner and evidence")
            for proof in row.get("evidence", []):
                if not isinstance(proof, dict) or proof.get("blob") != row.get("blob"):
                    errors.append(f"{path}: evidence does not bind current blob")
                elif not all(isinstance(proof.get(key), str) and proof[key]
                             for key in ("kind", "reference", "result")):
                    errors.append(f"{path}: evidence needs kind, reference, result")
    if not isinstance(ledger.get("removed_files", []), list):
        errors.append("removed_files must be an array")
    return errors


def review_snapshot(row, commit, reason):
    return {"snapshot_commit": commit, "reason": reason,
            **{key: copy.deepcopy(row.get(key)) for key in
               ("blob", "git_mode", "git_type", "status", "review_owner", "evidence")}}


def summarize(rows):
    result = {"total": len(rows)}
    for key in ("category", "phase", "status"):
        counts = {}
        for row in rows:
            value = row[key]
            counts[value] = counts.get(value, 0) + 1
        result[key] = dict(sorted(counts.items()))
    return result


def generate(repo, ref, previous=None):
    commit, entries = snapshot(repo, ref)
    previous_by_path = {row["path"]: row for row in previous["files"]} if previous else {}
    previous_commit = previous["snapshot_commit"] if previous else None
    rows = []
    for entry in entries:
        row = {**entry, **classify(entry["path"], entry["git_type"]),
               "status": "inventariado", "review_owner": None,
               "evidence": [], "history": []}
        old = previous_by_path.pop(entry["path"], None)
        if old:
            row["history"] = copy.deepcopy(old["history"])
            if signature(old) == signature(entry):
                for key in ("status", "review_owner", "evidence"):
                    row[key] = copy.deepcopy(old[key])
            else:
                row["history"].append(review_snapshot(old, previous_commit, "object_or_mode_changed"))
        rows.append(row)
    removed = copy.deepcopy(previous.get("removed_files", [])) if previous else []
    for path, old in sorted(previous_by_path.items()):
        removed.append({"path": path, "removed_at_snapshot": commit,
                        "history": copy.deepcopy(old["history"]) +
                        [review_snapshot(old, previous_commit, "removed_from_tree")]})
    return {"schema_version": SCHEMA_VERSION, "snapshot_commit": commit,
            "snapshot_commit_time": git(repo, "show", "-s", "--format=%cI", commit).decode().strip(),
            "scope": "Every tracked leaf entry in git ls-tree -r at snapshot_commit; no working-tree files.",
            "classification": "Heuristic proposed routing; owners are roles, not accepted assignments.",
            "verification": "Inventoried is not reviewed. check validates metadata, not the truth of review evidence.",
            "summary": summarize(rows), "files": rows, "removed_files": removed}


def check(repo, ledger):
    errors = validate_structure(ledger)
    if errors:
        return errors
    commit, entries = snapshot(repo, ledger["snapshot_commit"])
    if commit != ledger["snapshot_commit"]:
        errors.append("snapshot_commit must be the full immutable commit ID")
    expected = {row["path"]: row for row in entries}
    actual = {row["path"]: row for row in ledger["files"]}
    for path in sorted(expected.keys() - actual.keys()):
        errors.append(f"missing path: {path}")
    for path in sorted(actual.keys() - expected.keys()):
        errors.append(f"unexpected path: {path}")
    for path in sorted(expected.keys() & actual.keys()):
        if signature(expected[path]) != signature(actual[path]):
            errors.append(f"object/mode mismatch: {path}")
    if ledger.get("summary") != summarize(ledger["files"]):
        errors.append("summary does not match file rows")
    return errors


def delta(repo, ledger, ref):
    commit, entries = snapshot(repo, ref)
    old = {row["path"]: row for row in ledger["files"]}
    new = {row["path"]: row for row in entries}
    changed = []
    for path in sorted(old.keys() & new.keys()):
        if signature(old[path]) != signature(new[path]):
            changed.append({"path": path, "old_blob": old[path]["blob"],
                            "new_blob": new[path]["blob"],
                            "mode_changed": old[path]["git_mode"] != new[path]["git_mode"],
                            "previous_status": old[path]["status"], "review_invalidated": True})
    return {"from_commit": ledger["snapshot_commit"], "to_commit": commit,
            "added": sorted(new.keys() - old.keys()), "removed": sorted(old.keys() - new.keys()),
            "changed": changed,
            "unchanged_count": len(old.keys() & new.keys()) - len(changed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("generate")
    build.add_argument("--ref", required=True, help="Commit/ref to resolve and freeze in output")
    build.add_argument("--output", required=True)
    build.add_argument("--previous")
    validate = commands.add_parser("check")
    validate.add_argument("--ledger", required=True)
    compare = commands.add_parser("delta")
    compare.add_argument("--ledger", required=True)
    compare.add_argument("--ref", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            previous = load_ledger(args.previous) if args.previous else None
            if previous:
                errors = check(args.repo, previous)
                if errors:
                    raise ValueError("Invalid previous snapshot: " + "; ".join(errors))
            ledger = generate(args.repo, args.ref, previous)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(ledger, stream, indent=2, ensure_ascii=True)
                stream.write("\n")
            print(json.dumps({"output": str(output), "snapshot_commit": ledger["snapshot_commit"],
                              "summary": ledger["summary"]}, ensure_ascii=True))
        elif args.command == "check":
            ledger = load_ledger(args.ledger)
            errors = check(args.repo, ledger)
            print(json.dumps({"ok": not errors, "snapshot_commit": ledger["snapshot_commit"],
                              "files": len(ledger["files"]), "errors": errors}, ensure_ascii=True))
            return 1 if errors else 0
        else:
            ledger = load_ledger(args.ledger)
            errors = check(args.repo, ledger)
            if errors:
                raise ValueError("Invalid source snapshot: " + "; ".join(errors))
            print(json.dumps(delta(args.repo, ledger, args.ref), indent=2, ensure_ascii=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
