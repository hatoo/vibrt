"""Deletion-boundary and retention tests using only temporary, owned artifacts."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from auto_optimize_cleanup import cleanup_artifacts, UnsafeCleanupPath


class CleanupTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / "target/cleanup-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="cleanup space ", dir=test_root)
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name).resolve()
        self.output = self.repo / "target/auto-optimize"
        self.runs = self.output / "worktree/target/auto-optimize/runs"
        self.runs.mkdir(parents=True)

    def write(self, path, value="test payload"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path

    def metadata(self, path, value):
        return self.write(path, json.dumps(value))

    def make_run(self, number, status="rejected", finished=True, legacy=False):
        name = f"20260101T000000.{number:06d}Z"
        directory = (self.output / "runs" if legacy else self.runs) / name
        state = dict(run_id=name, directory=str(directory), status=status)
        if finished:
            state["finished_at"] = "2026-01-01T00:00:01+00:00"
        self.metadata(directory / "state.json", state)
        for path in ("build/release/native.dll", "build/debug/incremental/cache.bin",
                     "build/.rustc_info.json", "baseline/native.pyd", "inputs/mesh.npz",
                     "render/pixels.npy", "cache/compiled.ptx"):
            self.write(directory / path)
        for path in ("REPORT.md", "before.patch", "results.json", "render/preview.png",
                     "build.py", "memory-before.md", "baseline/source/fixture.npy",
                     "baseline/source/native.dll", "candidate/source/renderer.rs"):
            self.write(directory / path, "retain this evidence")
        return directory

    def junction(self, link, target):
        if os.name == "nt":
            # Both paths are generated under this test's verified temporary root.
            self.assertTrue(link.absolute().is_relative_to(self.repo))
            self.assertTrue(target.resolve().is_relative_to(self.repo))
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           check=True, capture_output=True, timeout=10)
        else:
            link.symlink_to(target, target_is_directory=True)

    def test_prunes_old_payloads_and_preserves_source_and_evidence(self):
        old = self.make_run(1)
        accepted = self.make_run(2, "accepted")
        recent = [self.make_run(i) for i in (3, 4, 5)]
        self.metadata(self.output / "latest.json", {"directory": str(recent[-1])})
        report = cleanup_artifacts(self.repo)
        self.assertEqual(report["removed_files"], 7)
        self.assertGreater(report["removed_bytes"], 0)
        self.assertFalse((old / "build").exists())
        self.assertFalse((old / "baseline/native.pyd").exists())
        self.assertFalse((old / "inputs/mesh.npz").exists())
        for path in ("REPORT.md", "before.patch", "results.json", "render/preview.png",
                     "build.py", "memory-before.md", "baseline/source/fixture.npy",
                     "baseline/source/native.dll", "candidate/source/renderer.rs"):
            self.assertEqual((old / path).read_text(), "retain this evidence")
        for directory in [accepted, *recent]:
            self.assertTrue((directory / "build/release/native.dll").is_file())
        self.assertTrue((old / "cleanup.json").is_file())
        self.assertTrue((self.output / "cleanup.json").is_file())
        self.assertEqual(cleanup_artifacts(self.repo)["removed_bytes"], 0)

    def test_latest_pending_pinned_running_and_unknown_rounds_are_preserved(self):
        pending = self.make_run(1, "failed")
        pinned = self.make_run(2)
        self.write(pinned / "KEEP_ARTIFACTS")
        running = self.make_run(3, "running", finished=False)
        malformed = self.make_run(4)
        self.write(malformed / "state.json", "invalid JSON")
        latest = self.make_run(5, "interrupted")
        eligible = self.make_run(6)
        newest = self.make_run(7)
        self.metadata(self.output / "latest.json", {"directory": str(latest)})
        self.metadata(self.output / "pending-push.json", {"source_run": str(pending)})
        cleanup_artifacts(self.repo, keep_runs=1)
        for directory in (pending, pinned, running, malformed, latest, newest):
            self.assertTrue((directory / "build/release/native.dll").is_file())
        self.assertFalse((eligible / "build").exists())

    def test_preview_does_not_remove_files_or_create_manifests(self):
        old = self.make_run(1)
        self.make_run(2)
        report = cleanup_artifacts(self.repo, keep_runs=1, dry_run=True)
        self.assertGreater(report["planned_bytes"], 0)
        self.assertEqual(report["removed_bytes"], 0)
        self.assertTrue((old / "build/release/native.dll").is_file())
        self.assertFalse((old / "cleanup.json").exists())
        self.assertFalse((self.output / "cleanup.json").exists())

    def test_junction_inside_cache_is_not_followed(self):
        old = self.make_run(1)
        self.make_run(2)
        important = self.write(self.repo / "unrelated/important.bin", "do not delete")
        self.junction(old / "build/linked-directory", important.parent)
        report = cleanup_artifacts(self.repo, keep_runs=1)
        self.assertEqual(important.read_text(), "do not delete")
        self.assertEqual(len(report["skipped_links"]), 1)
        self.assertFalse((old / "build/release/native.dll").exists())

    def test_linked_run_is_skipped_even_with_valid_looking_metadata(self):
        unrelated = self.repo / "other"
        unrelated.mkdir()
        link = self.runs / "20260101T000000.000001Z"
        self.junction(link, unrelated)
        self.metadata(unrelated / "state.json", dict(
            run_id=link.name, directory=str(link), status="rejected", finished_at="done"))
        important = self.write(unrelated / "build/important.dll")
        self.make_run(2)
        cleanup_artifacts(self.repo, keep_runs=1)
        self.assertTrue(important.is_file())

    def test_linked_managed_root_is_rejected_before_any_deletion(self):
        old = self.make_run(1)
        other = self.repo / "other-runs"
        other.mkdir()
        self.junction(self.output / "runs", other)
        with self.assertRaises(UnsafeCleanupPath):
            cleanup_artifacts(self.repo, keep_runs=1)
        self.assertTrue((old / "build/release/native.dll").is_file())

    def test_invalid_protection_metadata_fails_closed(self):
        old = self.make_run(1)
        self.make_run(2)
        self.write(self.output / "pending-push.json", "{")
        with self.assertRaises(ValueError):
            cleanup_artifacts(self.repo, keep_runs=1)
        self.assertTrue((old / "build/release/native.dll").is_file())
        self.metadata(self.output / "pending-push.json", {"source_run": str(self.repo / "unrelated")})
        with self.assertRaises(ValueError):
            cleanup_artifacts(self.repo, keep_runs=1)
        self.assertTrue((old / "build/release/native.dll").is_file())

    def test_legacy_runs_are_retired_but_repository_builds_are_untouched(self):
        old = self.make_run(1, legacy=True)
        self.make_run(2)
        manual_build = self.write(self.repo / "target/release/user.dll")
        scene = self.write(self.repo / "test_scenes/example.blend")
        cleanup_artifacts(self.repo, keep_runs=1)
        self.assertFalse((old / "build").exists())
        self.assertTrue(manual_build.is_file())
        self.assertTrue(scene.is_file())


if __name__ == "__main__":
    unittest.main()
