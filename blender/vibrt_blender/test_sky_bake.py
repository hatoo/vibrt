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
