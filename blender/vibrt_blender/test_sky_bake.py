"""Worker failure/cancellation tests; run without Blender."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("vibrt_sky_bake", Path(__file__).with_name("sky_bake.py"))
sky_bake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sky_bake)


class SkyBakeWorkerTests(unittest.TestCase):
    def test_worker_allows_export_before_waiting(self):
        # The child cannot finish until the caller does work inside the
        # context. A blocking start would fail the child's bounded wait.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            command = [
                sys.executable, "-c",
                "import pathlib, time, sys\n"
                "gate = pathlib.Path(sys.argv[1])\n"
                "deadline = time.monotonic() + 5\n"
                "while not gate.exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
                "assert gate.exists(), 'export did not overlap the worker'\n"
                "print('sky ready')",
                str(directory / "exported"),
            ]
            with sky_bake._start_worker(command, directory, lambda: False) as wait:
                (directory / "exported").touch()
                self.assertIn("sky ready", wait())

    def test_export_error_reaps_worker_without_waiting(self):
        processes = []
        popen = subprocess.Popen

        def spawn(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(sky_bake.subprocess, "Popen", side_effect=spawn):
                with self.assertRaisesRegex(RuntimeError, "export failed"):
                    with sky_bake._start_worker(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        Path(temporary), lambda: False,
                    ):
                        raise RuntimeError("export failed")
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())

    def test_worker_failure_includes_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "missing sky asset"):
                sky_bake._run_worker(
                    [sys.executable, "-c", "print('missing sky asset'); raise SystemExit(7)"],
                    Path(temporary), lambda: False,
                )

    def test_cancelling_render_stops_and_reaps_worker(self):
        processes = []
        popen = subprocess.Popen

        def spawn(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(sky_bake.subprocess, "Popen", side_effect=spawn):
                with self.assertRaises(KeyboardInterrupt):
                    sky_bake._run_worker(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        Path(temporary), lambda: True,
                    )
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())


if __name__ == "__main__":
    unittest.main()
