"""Preserve usize-compatible T-1 thread formats using the existing Cargo stub."""

import importlib.util
from pathlib import Path
import sys
import unittest


_HARNESS_PATH = Path(__file__).with_name("test_t1_oraculo.py")
_SPEC = importlib.util.spec_from_file_location("t1_thread_fixture_harness", _HARNESS_PATH)
harness = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = harness
_SPEC.loader.exec_module(harness)


@unittest.skipUnless(harness.BASH and Path(harness.BASH).is_file(), "Bash is required")
class T1ThreadFormatTests(unittest.TestCase):
    def test_usize_thread_formats_are_preserved_for_argument_and_environment(self):
        for source, value in (("argument", "02"), ("argument", "+02"),
                              ("environment", "02"), ("environment", "+02")):
            with self.subTest(source=source, value=value):
                fixture = harness.T1OraculoShellTests(
                    methodName="test_defaults_use_standard_target_and_serial_resources")
                try:
                    fixture.setUp()
                    overrides = {"CARGO_BUILD_JOBS": "1"}
                    if source == "argument":
                        result = fixture.run_script(value, overrides=overrides)
                    else:
                        overrides["T1_THREADS"] = value
                        result = fixture.run_script(overrides=overrides)
                    self.assertEqual(0, result.returncode, result.stdout + result.stderr)
                    environment = fixture.assert_cargo_invocation(result)
                    self.assertEqual(value, environment["T1_THREADS"])
                    self.assertEqual("1", environment["CARGO_BUILD_JOBS"])
                finally:
                    fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
