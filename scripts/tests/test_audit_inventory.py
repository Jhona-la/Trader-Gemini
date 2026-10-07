"""Contract tests use a temporary Git repository, never the trading workspace."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "audit_inventory.py"
SPEC = importlib.util.spec_from_file_location("audit_inventory", TOOL)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class AuditInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audit-inventory-test-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "audit-test@example.invalid")
        self.git("config", "user.name", "Audit Fixture")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.autocrlf", "false")
        self.paths = ["crates/risk-engine/src/ruin.rs", "crates/risk-engine/tests/contract.rs",
                      "docs/space name.md", "config_dir/model.json", "graphify-out/cache.json",
                      "data/fixture.csv"]
        for path in self.paths:
            self.write(path, "fixture\n")
        self.commit("baseline")
        self.baseline = self.git("rev-parse", "HEAD").strip()

    def git(self, *args):
        result = subprocess.run(["git", "-C", str(self.repo), *args],
                                capture_output=True, text=True, check=True)
        return result.stdout

    def write(self, path, text):
        output = self.repo / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")

    def commit(self, message):
        # This repository is a disposable fixture wholly owned by this test.
        for path in self.paths:
            if (self.repo / path).exists():
                self.git("add", "--", path)
        self.git("commit", "-qm", message)

    def ledger(self):
        return inventory.generate(self.repo, self.baseline)

    def verify_row(self, ledger, path):
        row = next(row for row in ledger["files"] if row["path"] == path)
        row["status"] = "verificado"
        row["review_owner"] = "fixture_reviewer"
        row["evidence"] = [{"blob": row["blob"], "kind": "behavior_test",
                            "reference": "fixture proof", "result": "pass"}]
        ledger["summary"] = inventory.summarize(ledger["files"])
        return row

    def test_complete_reproducible_snapshot_excludes_untracked_and_dirty_content(self):
        before = self.ledger()
        self.write("untracked.txt", "do not include")
        self.write(self.paths[0], "dirty source must not alter committed fingerprint")
        after = self.ledger()
        self.assertEqual(before, after)
        self.assertEqual(set(self.paths), {row["path"] for row in after["files"]})
        self.assertTrue(all(row["status"] == "inventariado" for row in after["files"]))
        self.assertEqual([], inventory.check(self.repo, after))
        categories = {row["category"] for row in after["files"]}
        self.assertEqual({"source", "test", "documentation", "configuration", "generated", "data_or_model"}, categories)

    def test_check_rejects_duplicate_missing_extra_and_wrong_hash(self):
        good = self.ledger()
        duplicate = copy.deepcopy(good)
        duplicate["files"].append(copy.deepcopy(duplicate["files"][0]))
        self.assertTrue(any("duplicate" in error for error in inventory.check(self.repo, duplicate)))
        missing = copy.deepcopy(good)
        missing["files"].pop()
        self.assertTrue(any("missing path" in error for error in inventory.check(self.repo, missing)))
        extra = copy.deepcopy(good)
        extra["files"].append({**copy.deepcopy(extra["files"][0]), "path": "invented.rs"})
        self.assertTrue(any("unexpected path" in error for error in inventory.check(self.repo, extra)))
        wrong = copy.deepcopy(good)
        wrong["files"][0]["blob"] = "0" * 40
        self.assertTrue(any("mismatch" in error for error in inventory.check(self.repo, wrong)))

    def test_content_change_resets_review_preserves_history_and_other_reviews(self):
        old = self.ledger()
        changed_path, unchanged_path = self.paths[:2]
        changed_old = copy.deepcopy(self.verify_row(old, changed_path))
        self.verify_row(old, unchanged_path)
        self.write(changed_path, "changed committed source\n")
        self.commit("change source")
        change = inventory.delta(self.repo, old, "HEAD")
        self.assertEqual([changed_path], [row["path"] for row in change["changed"]])
        new = inventory.generate(self.repo, "HEAD", old)
        changed = next(row for row in new["files"] if row["path"] == changed_path)
        unchanged = next(row for row in new["files"] if row["path"] == unchanged_path)
        self.assertEqual("inventariado", changed["status"])
        self.assertEqual([], changed["evidence"])
        self.assertEqual(changed_old["evidence"], changed["history"][0]["evidence"])
        self.assertEqual("verificado", unchanged["status"])
        self.assertEqual([], inventory.check(self.repo, new))

    def test_mode_only_change_invalidates_review_even_if_blob_unchanged(self):
        old = self.ledger()
        path = self.paths[0]
        self.verify_row(old, path)
        self.git("update-index", "--chmod=+x", "--", path)
        self.git("commit", "-qm", "executable bit")
        change = inventory.delta(self.repo, old, "HEAD")["changed"]
        self.assertEqual(1, len(change))
        self.assertTrue(change[0]["mode_changed"])
        self.assertEqual(change[0]["old_blob"], change[0]["new_blob"])
        new = inventory.generate(self.repo, "HEAD", old)
        self.assertEqual("inventariado", next(row for row in new["files"] if row["path"] == path)["status"])

    def test_deletion_keeps_review_and_new_file_starts_inventoried(self):
        old = self.ledger()
        path = self.paths[0]
        self.verify_row(old, path)
        self.git("rm", "--", path)
        added = "docs/new.md"
        self.paths.append(added)
        self.write(added, "new\n")
        self.commit("delete and add")
        change = inventory.delta(self.repo, old, "HEAD")
        self.assertEqual([path], change["removed"])
        self.assertEqual([added], change["added"])
        new = inventory.generate(self.repo, "HEAD", old)
        self.assertEqual(path, new["removed_files"][0]["path"])
        self.assertEqual("verificado", new["removed_files"][0]["history"][-1]["status"])
        self.assertEqual("inventariado", next(row for row in new["files"] if row["path"] == added)["status"])
        refreshed = inventory.generate(self.repo, "HEAD", new)
        self.assertEqual(new["removed_files"], refreshed["removed_files"])

    def test_verified_requires_owner_and_evidence_bound_to_current_object(self):
        ledger = self.ledger()
        row = ledger["files"][0]
        row["status"] = "verificado"
        self.assertTrue(inventory.validate_structure(ledger))
        self.verify_row(ledger, row["path"])
        self.assertEqual([], inventory.check(self.repo, ledger))
        row["evidence"][0]["blob"] = "0" * 40
        self.assertTrue(any("bind current blob" in error for error in inventory.validate_structure(ledger)))

    def test_cli_generate_check_delta_and_invalid_ledger_exit(self):
        output = self.repo / "ledger.json"
        commands = [
            ["generate", "--ref", self.baseline, "--output", str(output)],
            ["check", "--ledger", str(output)],
            ["delta", "--ledger", str(output), "--ref", self.baseline],
        ]
        for command in commands:
            result = subprocess.run([__import__("sys").executable, str(TOOL), "--repo", str(self.repo), *command],
                                    capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIsInstance(json.loads(result.stdout), dict)
        output.write_text('{"schema_version": 999}', encoding="utf-8")
        result = subprocess.run([__import__("sys").executable, str(TOOL), "--repo", str(self.repo),
                                 "check", "--ledger", str(output)], capture_output=True, text=True)
        self.assertNotEqual(0, result.returncode)
        self.assertIn("unsupported schema_version", result.stderr)


if __name__ == "__main__":
    unittest.main()
