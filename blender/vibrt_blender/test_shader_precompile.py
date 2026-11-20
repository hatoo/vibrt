"""Background shader job lifecycle checks, without Blender or a GPU."""

import importlib.util
from pathlib import Path
import sys
import time
import unittest

spec = importlib.util.spec_from_file_location(
    "vibrt_shader_precompile", Path(__file__).with_name("shader_precompile.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompileJobTests(unittest.TestCase):
    def test_cancel_reaps_worker_and_removes_snapshot(self):
        job = module.CompileJob()
        try:
            job.launch([sys.executable, "-c", "import time; time.sleep(30)"])
            self.assertFalse(job.done())
            job.cancel()
            self.assertTrue(job.cancelled)
            self.assertIsNotNone(job.process.poll())
            self.assertFalse(job.directory.exists())
        finally:
            job.close()

    def test_failure_reports_worker_diagnostic(self):
        job = module.CompileJob()
        try:
            job.launch(
                [sys.executable, "-c", "print('missing OptiX'); raise SystemExit(7)"]
            )
            job.process.wait(timeout=10)
            with self.assertRaisesRegex(RuntimeError, "missing OptiX"):
                job.result()
        finally:
            job.close()

    def test_result_is_published_only_after_worker_exit(self):
        job = module.CompileJob()
        try:
            job.launch(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys,time\n"
                    "p=Path(sys.argv[1]); (p/'result.json').write_text('{\"seconds\": 1.25}')\n"
                    "while not (p/'release').exists(): time.sleep(0.01)",
                    str(job.directory),
                ]
            )
            deadline = time.monotonic() + 10
            while not (job.directory / "result.json").exists():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            with self.assertRaisesRegex(RuntimeError, "still running"):
                job.result()
            (job.directory / "release").touch()
            job.process.wait(timeout=10)
            self.assertEqual(job.result(), {"seconds": 1.25})
        finally:
            job.close()


if __name__ == "__main__":
    unittest.main()
