"""Hosted-only causal LG22 experiment. --self-test is pure parser validation.

No local Cargo execution is supported. This runner never fetches, pushes,
creates branches, or changes a remote. The workflow supplies all Git objects.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

BASE = "28125c1ae71fb95b25dc25c4c78c1ecddb9d46a7"
CANDIDATE = "d2bf352c4ea2fa8e17758362b22dce5509248b51"
CI_REF = "refs/heads/codex/si-lg22-causal-ci-20261020"
TOOLCHAIN = "nightly-2026-06-30"
SOURCE = "crates/god-engine-core/src/resultado_orden.rs"
BASE_SHA256 = "3a11957d9bb70ae9dbd6331d72d01f946bbb615edea89609582249cac7cd7645"
CANDIDATE_SHA256 = "f12561728bc7cd64cb729f630c78d71889cde4e7755202ab322cd8b99427dd19"
CORPUS_SHA256 = "f2ff6af8c5b9915b2785f27762b329470174775062091c618f81b0eca6283c94"
BASE_BLOB = "57d8b3dafc911f138eb61b3683d55f1d4bd2566e"
CANDIDATE_BLOB = "f8a4a20ec16cd769e67d0b6a75556a1f60a9b1b8"
MARKER = b"#[cfg(test)]\nmod tests_lg22_numeric {"
FILTER = "resultado_orden::tests_lg22_numeric"
SHORT_NAMES = (
    "lg22_numeric_factores_invariantes_hasta_escalas_subnormales",
    "lg22_numeric_recursion_y_reinicio_conservan_la_escala",
    "lg22_numeric_cociente_intermedio_y_factor_enorme_guardan_log_finito",
    "lg22_numeric_resta_desbordada_conserva_factor_representable",
    "lg22_numeric_preserva_recorte_y_flags_fuera_de_garantia",
    "lg22_numeric_factor_positivo_junto_al_recorte_no_se_anula",
)
NAMES = tuple(FILTER + "::" + name for name in SHORT_NAMES)
RED_PASS = FILTER + "::lg22_numeric_preserva_recorte_y_flags_fuera_de_garantia"
RED_FAILS = set(NAMES) - {RED_PASS}
CHECK = ["cargo", "+" + TOOLCHAIN, "check", "--workspace", "--all-targets", "--locked"]
TEST = ["cargo", "+" + TOOLCHAIN, "test", "-p", "god-engine-core", "--lib", FILTER,
        "--locked", "--", "--test-threads=1", "--nocapture"]
REQUIRED_ENV = {"CARGO_BUILD_JOBS": "2", "CARGO_INCREMENTAL": "0",
                "CARGO_PROFILE_DEV_DEBUG": "0", "CARGO_PROFILE_TEST_DEBUG": "0"}
UNSET_ENV = ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUST_MIN_STACK", "CARGO_TARGET_DIR")


class InvalidEvidence(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def corpus_of(source):
    require(source.count(MARKER) == 1, "numeric module marker must occur exactly once")
    corpus = source[source.index(MARKER):]
    require(sha(corpus) == CORPUS_SHA256, "numeric corpus SHA mismatch")
    require(corpus.count(b"#[test]") == 6 and b"#[ignore" not in corpus,
            "corpus must have exactly six unignored tests")
    names = tuple(name.decode("ascii") for name in re.findall(
        rb"(?m)^    fn (lg22_numeric_[a-z_]+)\(", corpus))
    require(names == SHORT_NAMES, "numeric corpus test names/order changed")
    return corpus


def classify_test(raw, code, phase, module_lines):
    """Fail closed: a compiler failure/abort/zero tests is never causal RED."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw.decode("utf-8", "replace"))
    for forbidden in ("stack overflow", "overflowed its stack", "fatal runtime error",
                      "could not compile", "error[e", "signal: 6", "signal: 11"):
        require(forbidden not in text.lower(), "invalid test infrastructure: " + forbidden)
    require(len(re.findall(r"(?m)^running 6 tests\r?$", text)) == 1,
            "must observe exactly one libtest run of six tests")
    summaries = re.findall(
        r"(?m)^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
        r"(\d+) measured; (\d+) filtered out; finished in [^\r\n]+\r?$", text)
    require(len(summaries) == 1, "missing/ambiguous libtest completion summary")
    status, passed, failed, ignored, measured, filtered = summaries[0]
    passed, failed, ignored, measured, filtered = map(int, (passed, failed, ignored, measured, filtered))
    require(ignored == 0 and measured == 0 and passed + failed == 6,
            "unexpected ignored/measured/missing tests")
    starts = re.findall(r"(?m)^test (" + re.escape(FILTER) + r"::[a-z0-9_]+) \.\.\.", text)
    require(len(starts) == 6 and set(starts) == set(NAMES), "executed names differ from pinned corpus")
    panics = re.findall(
        r"thread '([^']+)'(?:\s+\([^\r\n]*?\))?\s+panicked at "
        r"([^\r\n]+):(\d+):(\d+):\r?\n([^\r\n]*)", text)
    if phase == "RED":
        require(code == 101 and status == "FAILED" and (passed, failed) == (1, 5),
                "RED requires Cargo exit 101 and observed 1 PASS / 5 FAIL")
        failed_names = re.findall(r"(?m)^    (" + re.escape(FILTER) + r"::[a-z0-9_]+)\r?$", text)
        require(len(failed_names) == 5 and set(failed_names) == RED_FAILS,
                "RED failure list differs from the five predicted numeric tests")
        require(re.search(r"(?m)^test " + re.escape(RED_PASS) + r" \.\.\. ok\r?$", text),
                "the preserved-flags test did not explicitly pass")
        require(len(panics) == 5 and {p[0] for p in panics} == RED_FAILS,
                "RED must contain exactly one assertion panic for each expected failing test")
        for name, path, line, column, message in panics:
            require(path.replace("\\", "/").endswith(SOURCE), "panic outside numeric test source")
            require(module_lines[0] <= int(line) <= module_lines[1], "panic outside appended module")
            require(message.startswith("assertion failed:"), "non-assertion panic is not numeric RED")
        observed_failed = sorted(failed_names)
    else:
        require(phase == "GREEN", "invalid phase")
        require(code == 0 and status == "ok" and (passed, failed) == (6, 0),
                "GREEN requires Cargo exit 0 and observed 6 PASS / 0 FAIL")
        require(not panics and "panicked at" not in text and "failures:" not in text,
                "unexpected panic/failure output in GREEN")
        oks = re.findall(r"(?m)^test (" + re.escape(FILTER) + r"::[a-z0-9_]+) \.\.\. ok\r?$", text)
        require(len(oks) == 6 and set(oks) == set(NAMES), "GREEN did not explicitly pass all six names")
        observed_failed = []
    return {"phase": phase, "observed": True, "passed": passed, "failed": failed,
            "ignored": ignored, "filtered_out": filtered, "failed_names": observed_failed}


def parser_self_test():
    """Synthetic text only: no subprocess, git, Cargo, checkout or CI."""
    def fixture(red):
        lines = ["running 6 tests"]
        for name in NAMES:
            lines.append("test " + name + " ... " + ("FAILED" if red and name in RED_FAILS else "ok"))
        if red:
            for name in sorted(RED_FAILS):
                lines.extend(["thread '" + name + "' (42) panicked at " + SOURCE + ":1001:9:",
                              "assertion failed: numeric_contract"])
            lines.extend(["failures:"] + ["    " + n for n in sorted(RED_FAILS)])
        lines.append("test result: " + ("FAILED. 1 passed; 5 failed" if red else "ok. 6 passed; 0 failed")
                     + "; 0 ignored; 0 measured; 2000 filtered out; finished in 0.01s")
        return ("\n".join(lines) + "\n").encode()
    red, green = fixture(True), fixture(False)
    classify_test(red, 101, "RED", (1000, 1200))
    classify_test(green, 0, "GREEN", (1000, 1200))
    classify_test(red.replace(b"\n", b"\r\n"), 101, "RED", (1000, 1200))
    classify_test(green.replace(b"\n", b"\r\n"), 0, "GREEN", (1000, 1200))
    bad = [
        (b"error[E0425]: helper missing\n", 101, "RED"),
        (b"thread 'x' has overflowed its stack\nfatal runtime error\n", 101, "RED"),
        (b"running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 6 filtered out; finished in 0.00s\n", 0, "GREEN"),
        (red, 0, "RED"), (green, 101, "GREEN"), (green, 0, "RED"),
        (red.replace(b"1 passed; 5 failed", b"2 passed; 4 failed"), 101, "RED"),
        (red.replace(b"assertion failed:", b"called Option::unwrap:"), 101, "RED"),
        (red.replace(b":1001:9:", b":10:9:"), 101, "RED"),
        (red.replace(SOURCE.encode(), b"src/unrelated.rs"), 101, "RED"),
        (red.replace(RED_PASS.encode(), b"unrelated::test"), 101, "RED"),
        (green.replace(b"6 passed; 0 failed; 0 ignored", b"5 passed; 0 failed; 1 ignored"), 0, "GREEN"),
    ]
    for raw, code, phase in bad:
        try:
            classify_test(raw, code, phase, (1000, 1200))
        except InvalidEvidence:
            continue
        raise AssertionError("invalid synthetic evidence accepted: " + phase)
    return {"scope": "pure synthetic parser validation, no Cargo/git/CI execution",
            "accepted_valid_fixtures": 4, "rejected_invalid_fixtures": len(bad)}


class Experiment:
    def __init__(self, repo, out, args):
        self.repo, self.out = repo, out
        self.report = {"schema": "si-lg22-causal-v1", "started_at": utc(), "status": "INCOMPLETE",
                       "base": BASE, "candidate": CANDIDATE, "source": SOURCE,
                       "source_sha256": CANDIDATE_SHA256, "corpus_sha256": CORPUS_SHA256,
                       "toolchain": TOOLCHAIN, "check_argv": CHECK, "test_argv": TEST,
                       "expected_red": {"passed": 1, "failed": 5, "observed": False},
                       "expected_green": {"passed": 6, "failed": 0, "observed": False},
                       "github": {k: os.environ.get(k) for k in (
                           "GITHUB_REPOSITORY", "GITHUB_SHA", "GITHUB_REF", "GITHUB_EVENT_NAME",
                           "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "RUNNER_OS", "RUNNER_ENVIRONMENT")},
                       "environment": {k: os.environ.get(k) for k in tuple(REQUIRED_ENV) + UNSET_ENV},
                       "python": sys.version, "platform": platform.platform(),
                       "standard_target_dir": str(repo / "target"),
                       "commands": [], "phases": {}, "runner_sha256": sha(Path(__file__).read_bytes()),
                       "expected_runner_sha256": args.expected_runner_sha256}
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "runner.py").write_bytes(Path(__file__).read_bytes())
        self.flush()

    def flush(self):
        (self.out / "receipt.json").write_text(json.dumps(self.report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def logged(self, argv, label):
        path = self.out / (label + ".log")
        record = {"argv": argv, "cwd": str(self.repo), "started_at": utc(),
                  "returncode": None, "log": path.name}
        self.report["commands"].append(record)
        self.flush()
        started = time.monotonic()
        print(json.dumps({"command": argv, "log": path.name}), flush=True)
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        with path.open("wb") as log:
            process = subprocess.Popen(argv, cwd=str(self.repo), env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            code = process.wait()
        raw = path.read_bytes()
        record.update(returncode=code, elapsed_seconds=round(time.monotonic() - started, 3),
                      sha256=sha(raw), bytes=len(raw), ended_at=utc())
        self.flush()
        return code, raw

    def git(self, *argv):
        result = subprocess.run(["git"] + list(argv), cwd=str(self.repo),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        require(result.returncode == 0, "git read failed: " + " ".join(argv)
                + "\n" + result.stderr.decode("utf-8", "replace"))
        return result.stdout

    def checkout(self, pin, phase):
        code, _ = self.logged(["git", "-c", "core.autocrlf=false", "checkout", "--detach", "--force", pin],
                              phase + "-checkout")
        require(code == 0, "checkout failed: " + phase)
        require(self.git("rev-parse", "HEAD").decode().strip() == pin, "checkout pin mismatch")

    def phase(self, name, pin, source, corpus, append_only):
        self.checkout(pin, name)
        path = self.repo / SOURCE
        checkout_sha = sha(path.read_bytes())
        # Git on Windows can materialize CRLF. Restore ONLY this source from
        # canonical git-show bytes; every other file remains the full checkout.
        path.write_bytes(source)
        require(path.read_bytes() == source, name + " canonical source write differs from git-show bytes")
        if append_only:
            require(MARKER not in source and b"fn log_factor(" not in source,
                    "BASE already contains numeric corpus/new helper")
            require(source.endswith(b"\n"), "BASE must end with LF")
            working = source + b"\n" + corpus
            require(working[:len(source)] == source, "BASE producer prefix changed")
            path.write_bytes(working)  # Only the exact new test module; producer bytes stay unchanged.
        else:
            working = source
        require(corpus_of(working) == corpus, "corpus differs between phases")
        start = working[:working.index(MARKER)].count(b"\n") + 1
        module_lines = (start, working.count(b"\n") + 1)
        (self.out / (name + "-source.rs")).write_bytes(working)
        self.report["phases"][name] = {"pin": pin, "source_sha256": sha(working),
                                      "head_before": self.git("rev-parse", "HEAD").decode().strip(),
                                      "checkout_source_sha256": checkout_sha,
                                      "canonical_producer_sha256": sha(source),
                                      "actual_source_sha256_before": sha(path.read_bytes()),
                                      "actual_corpus_sha256_before": sha(corpus_of(path.read_bytes())),
                                      "append_only": append_only, "module_lines": module_lines,
                                      "corpus_sha256": sha(corpus), "status": "INCOMPLETE"}
        self.flush()
        code, _ = self.logged(CHECK, name + "-check-all-targets")
        self.report["phases"][name].update(
            actual_source_sha256_after_check=sha(path.read_bytes()),
            head_after_check=self.git("rev-parse", "HEAD").decode().strip())
        self.flush()
        require(code == 0, name + " check --all-targets failed: infrastructure/compile, not numeric RED")
        require(path.read_bytes() == working, name + " source changed during cargo check")
        code, raw = self.logged(TEST, name + "-numeric-tests")
        after = path.read_bytes()
        head_after = self.git("rev-parse", "HEAD").decode().strip()
        after_corpus_sha = sha(after[after.index(MARKER):]) if after.count(MARKER) == 1 else None
        self.report["phases"][name].update(head_after=head_after,
                                          actual_source_sha256_after_tests=sha(after),
                                          actual_corpus_sha256_after_tests=after_corpus_sha)
        self.flush()
        observed = classify_test(raw, code, name, module_lines)
        require(after == working and corpus_of(after) == corpus,
                name + " source/corpus changed during tests")
        require(head_after == pin, name + " HEAD changed during check/tests")
        self.report["phases"][name].update(status="VALID", result=observed)
        self.flush()

    def execute(self, args):
        require(args.base == BASE and args.candidate == CANDIDATE, "literal pins differ from sealed runner")
        require(re.fullmatch(r"[0-9a-f]{64}", args.expected_runner_sha256 or ""), "invalid runner SHA argument")
        require(self.report["runner_sha256"] == args.expected_runner_sha256, "bootstrap runner SHA mismatch")
        require(self.git("status", "--porcelain").strip() == b"", "initial CI checkout is dirty")
        workflow_sha = self.git("rev-parse", "HEAD").decode().strip()
        require(workflow_sha == os.environ.get("GITHUB_SHA"), "initial checkout is not workflow GITHUB_SHA")
        self.report["workflow_commit"] = workflow_sha
        changed = self.git("diff", "--name-only", BASE, CANDIDATE).decode().splitlines()
        require(changed == [SOURCE], "pinned source candidate changes more than the reviewed atom")
        self.report["candidate_changed_paths"] = changed
        require(self.git("rev-parse", BASE + ":" + SOURCE).decode().strip() == BASE_BLOB, "BASE blob mismatch")
        require(self.git("rev-parse", CANDIDATE + ":" + SOURCE).decode().strip() == CANDIDATE_BLOB,
                "candidate blob mismatch")
        base = self.git("show", BASE + ":" + SOURCE)
        candidate = self.git("show", CANDIDATE + ":" + SOURCE)
        require(sha(base) == BASE_SHA256 and sha(candidate) == CANDIDATE_SHA256, "raw source SHA mismatch")
        corpus = corpus_of(candidate)
        locks = [self.git("show", pin + ":Cargo.lock") for pin in (BASE, CANDIDATE)]
        require(locks[0] == locks[1], "Cargo.lock differs between phases")
        self.report["cargo_lock_sha256"] = sha(locks[0])
        self.report["source_blobs"] = {"base": BASE_BLOB, "candidate": CANDIDATE_BLOB}
        self.report["base_raw_sha256"] = sha(base)
        (self.out / "base-original.rs").write_bytes(base)
        (self.out / "candidate-original.rs").write_bytes(candidate)
        (self.out / "numeric-corpus.rs").write_bytes(corpus)
        (self.out / "candidate.diff").write_bytes(self.git("diff", "--binary", BASE, CANDIDATE, "--", SOURCE))
        for argv, label in ((["cargo", "+" + TOOLCHAIN, "--version"], "cargo-version"),
                            (["rustc", "+" + TOOLCHAIN, "--version", "--verbose"], "rustc-version")):
            code, _ = self.logged(argv, label)
            require(code == 0, "toolchain version command failed")
        self.phase("RED", BASE, base, corpus, True)
        self.phase("GREEN", CANDIDATE, candidate, corpus, False)
        self.report["status"] = "CAUSAL_RED_GREEN_CONFIRMED"
        self.report["scope"] = "six pinned numeric contracts only; not full runtime, economics or all IEEE-754"

    def finish(self):
        self.report["ended_at"] = utc()
        self.report["artifacts"] = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                                    for p in sorted(self.out.iterdir()) if p.is_file()
                                    and p.name not in ("receipt.json", "receipt.sha256")]
        self.flush()
        digest = sha((self.out / "receipt.json").read_bytes())
        (self.out / "receipt.sha256").write_text(digest + "  receipt.json\n", encoding="ascii")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write("\nLG22 causal result: **" + self.report["status"] + "**.\n\n"
                             + "BASE `" + BASE + "`; candidate `" + CANDIDATE + "`.\n\n"
                             + "Receipt SHA-256: `" + digest + "`. Expected counts are not observations.\n")
        print(json.dumps({"status": self.report["status"], "receipt_sha256": digest}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--out-dir")
    parser.add_argument("--base")
    parser.add_argument("--candidate")
    parser.add_argument("--expected-runner-sha256")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(parser_self_test(), sort_keys=True))
        return 0
    require(args.repo and args.out_dir, "repo/out-dir required")
    # This guard runs BEFORE any subprocess or Git/Cargo action.
    require(os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"
            and os.environ.get("RUNNER_OS") == "Windows" and platform.system() == "Windows",
            "refusing non-hosted/non-Windows execution; local Cargo is prohibited")
    require(os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_REF") == CI_REF,
            "isolated CI push branch required; no PR/main/manual event")
    for key, value in REQUIRED_ENV.items():
        require(os.environ.get(key) == value, "unexpected environment: " + key)
    for key in UNSET_ENV:
        require(not os.environ.get(key), key + " must remain unset (no flag/stack/target override)")
    repo, out = Path(args.repo).resolve(), Path(args.out_dir).resolve()
    require(repo == Path(os.environ.get("GITHUB_WORKSPACE", "")).resolve(), "repo is not GITHUB_WORKSPACE")
    require(repo != out and repo not in out.parents, "evidence must remain outside Git checkout")
    require(repo not in Path(__file__).resolve().parents, "bootstrap script must persist outside checkout")
    experiment = Experiment(repo, out, args)
    try:
        experiment.execute(args)
        return_code = 0
    except Exception as error:
        experiment.report.update(status="INVALID_CAUSAL_EVIDENCE", error_type=type(error).__name__, error=str(error))
        (out / "runner-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        return_code = 1
    finally:
        experiment.finish()
    return return_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InvalidEvidence as error:
        print("PRECONDITION REJECTED: " + str(error), file=sys.stderr)
        sys.exit(2)
