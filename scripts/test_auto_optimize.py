"""Exercise the loop with a fake CLI, without API requests or GPU rendering."""
from contextlib import redirect_stdout
import ctypes
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import auto_optimize
import auto_optimize_git


RUNNER = Path(auto_optimize.__file__).resolve()
REAL_SLEEP = time.sleep
FAKE_CODEX = r'''
import json
from pathlib import Path
import subprocess
import sys
import time

args = sys.argv[1:]
repo = Path(args[args.index("--cd") + 1])
result_path = Path(args[args.index("--output-last-message") + 1])
control = Path(__file__).parent
output = control / "target/auto-optimize"
prompt = sys.stdin.buffer.read().decode("utf-8")
counter = control / "fake_count.txt"
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
(result_path.parent / "received.json").write_text(
    json.dumps({"args": args, "prompt": prompt}), encoding="utf-8")
print(json.dumps({"type": "test.round", "count": count}), flush=True)
print("fake stderr", file=sys.stderr, flush=True)
mode = (control / "fake_mode.txt").read_text()
if mode in {"probe_success", "probe_blocked"}:
    if mode == "probe_success":
        for name in ("shell-probe.txt", "nested/copy.txt", "patch-probe.txt"):
            path = result_path.parent / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("codex-sandbox-write-ok", encoding="utf-8")
    result_path.write_text(json.dumps(dict(
        status="no_change" if mode == "probe_success" else "blocked", summary=mode,
        changes=[], measurements=[], validation=[], next_step="", risks=[], commit_message="")))
    sys.exit(0)
if mode == "failures" and count == 1:
    sys.exit(7)
if mode == "failures" and count == 2:
    result_path.write_text('{"status": "accepted"}', encoding="utf-8")
    sys.exit(0)
if mode == "hold":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (control / "pids.json").write_text(
        json.dumps([__import__("os").getpid(), child.pid]), encoding="utf-8")
    try:
        while not (control / "RELEASE").exists():
            time.sleep(0.05)
    finally:
        child.terminate()
        child.wait()
if mode == "stop":
    (output / "STOP").touch()
(repo / "scene.txt").write_text(f"optimized round {count}\n", encoding="utf-8", newline="\n")
if mode != "missing_report":
    (result_path.parent / "REPORT.md").write_text("Fake benchmark report for supervisor tests.\n")
result = dict(status="accepted", summary=f"Round {count} completed",
              changes=["scene.txt"], measurements=["Fake comparison: 10 ms -> 9 ms"],
              validation=["fake CLI"], commit_message=f"renderer: improve fixture round {count}",
              next_step=f"Continue after round {count}", risks=[])
if mode == "rejected":
    result.update(status="rejected", commit_message="")
result_path.write_text(json.dumps(result), encoding="utf-8")
'''


class LocalPullRequests:
    """Model the server-side merge using only the test's bare remote."""
    repository = "fixture/renderer"

    def __init__(self, workspace):
        self.remote = Path(auto_optimize.git(workspace, "remote", "get-url", "origin").decode().strip())
        self.state = self.remote / "pull.json"

    def pull(self, branch, result, message_path):
        head = auto_optimize_git.revision(self.remote, "refs/heads/" + branch)
        pr = dict(number=1, head=dict(sha=head, ref=branch, repo=dict(full_name=self.repository)),
                  base=dict(ref="master"), html_url="fixture/pr", state="open", merged=False)
        self.state.write_text(json.dumps(pr))
        return pr

    def api(self, path):
        return json.loads(self.state.read_text())

    def checks(self, head):
        marker = self.remote / "fail-ci-once"
        if marker.exists():
            marker.unlink()
            return False, [{"name": "Renderer checks", "output": "deslop-please failed"}]
        return True, []

    def merge(self, number, head):
        if not auto_optimize_git.ancestor(self.remote, "master", head):
            raise auto_optimize_git.PublishBlocked("Fixture merge conflict")
        auto_optimize.git(self.remote, "update-ref", "refs/heads/master", head)
        pr = self.api("pulls/1")
        pr.update(merged=True, state="closed")
        self.state.write_text(json.dumps(pr))


class AutoOptimizeTests(unittest.TestCase):
    def setUp(self):
        test_root = RUNNER.parents[1] / "target/auto-optimize-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="loop 日本語 & ", dir=test_root)
        self.repo = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.fake = self.repo / "fake codex.py"
        self.fake.write_text(FAKE_CODEX, encoding="utf-8")
        self.set_mode("success")
        self.git("init", "--quiet", "--initial-branch=master")
        self.git("config", "user.name", "Optimizer Test")
        self.git("config", "user.email", "optimizer@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        (self.repo / ".gitignore").write_text("/target\n", encoding="utf-8", newline="\n")
        (self.repo / "scene.txt").write_text("baseline\n", encoding="utf-8", newline="\n")
        self.git("add", ".gitignore", "scene.txt")
        self.git("-c", "user.name=Optimizer Test", "-c", "user.email=optimizer@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "test baseline")
        self.base = self.git("rev-parse", "HEAD").decode().strip()
        self.remote = self.repo / "remote.git"
        self.git("init", "--bare", "--quiet", "--initial-branch=master", str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "--quiet", "origin", "master")
        (self.repo / "scene.txt").write_text("staged user change\n", encoding="utf-8", newline="\n")
        self.git("add", "scene.txt")
        (self.repo / "scene.txt").write_text("existing user change\n", encoding="utf-8")
        (self.repo / "keep.txt").write_text("existing untracked file\n", encoding="utf-8")
        self.output = self.repo / "target/auto-optimize"

    def git(self, *args):
        return auto_optimize.git(self.repo, *args)

    def set_mode(self, mode):
        (self.repo / "fake_mode.txt").write_text(mode, encoding="utf-8")

    def arguments(self, *args):
        return ["--repo", str(self.repo), "--codex", str(self.fake),
                "--pause-seconds", "0", "--retry-seconds", "0", *args]

    def invoke(self, *args):
        return subprocess.run([sys.executable, __file__, "--fixture-runner", *self.arguments(*args)],
                              capture_output=True, timeout=60)

    def remote_head(self):
        return auto_optimize.git(self.remote, "rev-parse", "refs/heads/master").decode().strip()

    def hook(self, directory, name, body):
        path = directory / "hooks" / name
        path.write_text("#!/bin/sh\n" + body, encoding="utf-8", newline="\n")
        path.chmod(0o755)
        return path

    def rounds(self):
        return sorted((self.output / "worktree/target/auto-optimize/runs").iterdir())

    def read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def wait_for(self, path, process=None):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if path.exists() and path.stat().st_size:
                return
            if process is not None and process.poll() is not None:
                self.fail(f"Runner exited early: {process.communicate()}")
            REAL_SLEEP(0.05)
        self.fail(f"Timed out waiting for {path}")

    def test_rounds_pass_context_and_preserve_existing_work(self):
        completed = self.invoke("--iterations", "2")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rounds = self.rounds()
        self.assertEqual(len(rounds), 2)
        received = self.read_json(rounds[1] / "received.json")
        self.assertIn("Continue after round 1", received["prompt"])
        self.assertIn(str(self.repo), received["prompt"])
        self.assertNotIn("{{REPO}}", received["prompt"])
        self.assertEqual(received["args"][-1], "-")
        self.assertIn("--no-daemon", received["args"])
        self.assertNotIn("--add-dir", received["args"])
        workspace = self.output / "worktree"
        for flag in ("--output-schema", "--output-last-message"):
            self.assertTrue(Path(received["args"][received["args"].index(flag) + 1]).is_relative_to(workspace))
        self.assertIn(str(workspace / "target/auto-optimize/MEMORY.md"), received["prompt"])
        self.assertIn(str(rounds[1]), received["prompt"])
        self.assertEqual((self.repo / "scene.txt").read_text(), "existing user change\n")
        self.assertEqual((self.repo / "keep.txt").read_text(), "existing untracked file\n")
        self.assertEqual(self.git("show", ":scene.txt"), b"staged user change\n")
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.base)
        for directory in rounds:
            self.assertEqual((directory / "before.patch").read_bytes(), b"")
            self.assertEqual((directory / "before.patch").read_bytes(),
                             (directory / "after.patch").read_bytes())
            self.assertTrue((directory / "events.jsonl").stat().st_size)
            self.assertIn("fake stderr", (directory / "stderr.log").read_text())
        state = self.read_json(self.output / "latest.json")
        self.assertEqual(state["status"], "accepted")
        self.assertEqual(state["publication"]["commit"], self.remote_head())
        self.assertEqual(auto_optimize.git(self.remote, "show", "master:scene.txt"), b"optimized round 2\n")
        self.assertEqual(auto_optimize.git(self.remote, "ls-tree", "--name-only", "master").splitlines(),
                         [b".gitignore", b"scene.txt"])
        self.assertIn(b"Fake comparison", auto_optimize.git(self.remote, "log", "-1", "--format=%B", "master"))
        self.assertFalse((self.output / "pending-push.json").exists())

    def test_artifacts_outside_agent_workspace_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside its working directory"):
            auto_optimize.command(["codex"], self.output / "worktree", self.output / "runs/example")

    def test_sandbox_check_requires_real_probe_files_and_never_publishes(self):
        self.set_mode("probe_blocked")
        blocked = self.invoke("--check-sandbox")
        self.assertEqual(blocked.returncode, 1, blocked.stderr)
        self.assertEqual(self.read_json(self.output / "sandbox-check.json")["status"], "blocked")
        self.set_mode("probe_success")
        completed = self.invoke("--check-sandbox")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        check = self.read_json(self.output / "sandbox-check.json")
        self.assertEqual(check["status"], "passed")
        self.assertTrue(all(check["checks"].values()))
        self.assertEqual(self.remote_head(), self.base)
        self.assertFalse((self.output / "pending-push.json").exists())
        self.assertFalse((self.output / "latest.json").exists())

    def test_legacy_memory_is_imported_once_without_overwriting_new_notes(self):
        self.output.mkdir(parents=True)
        legacy = self.output / "MEMORY.md"
        legacy.write_text("Earlier validated ideas.\n", encoding="utf-8")
        workspace = self.output / "worktree"
        directory, memory = auto_optimize.round_storage(workspace, self.output, "first")
        self.assertTrue(directory.is_relative_to(workspace))
        self.assertIn("Earlier validated ideas", memory.read_text(encoding="utf-8"))
        self.assertIn(str(legacy), memory.read_text(encoding="utf-8"))
        memory.write_text("Newer experiment notes.\n", encoding="utf-8")
        auto_optimize.round_storage(workspace, self.output, "second")
        self.assertEqual(memory.read_text(encoding="utf-8"), "Newer experiment notes.\n")
        self.assertEqual(legacy.read_text(encoding="utf-8"), "Earlier validated ideas.\n")

    def test_retries_cli_failures_and_invalid_results(self):
        self.set_mode("failures")
        completed = self.invoke("--iterations", "3")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        states = [self.read_json(path / "state.json") for path in self.rounds()]
        self.assertEqual([state["status"] for state in states], ["failed", "failed", "accepted"])
        self.assertEqual(states[0]["exit_code"], 7)
        self.assertIn("incomplete round result", states[1]["error"])
        self.assertIn('"status": "failed"',
                      self.read_json(self.rounds()[2] / "received.json")["prompt"])

    def test_low_disk_space_blocks_before_starting_codex(self):
        completed = self.invoke("--iterations", "1", "--min-free-gib", "1000000000")
        self.assertEqual(completed.returncode, 1, completed.stdout)
        state = self.read_json(self.output / "latest.json")
        self.assertEqual(state["status"], "blocked")
        self.assertIn("free after cleanup", state["error"])
        self.assertFalse((self.repo / "fake_count.txt").exists())
        self.assertEqual(self.remote_head(), self.base)

    def test_default_unlimited_loop_obeys_stop_after_completed_round(self):
        self.set_mode("stop")
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(self.rounds()), 1)
        self.assertEqual(self.read_json(self.output / "latest.json")["status"], "accepted")
        self.assertTrue((self.rounds()[0] / "result.json").is_file())

    def test_rejected_and_missing_evidence_are_not_published(self):
        self.set_mode("rejected")
        self.assertEqual(self.invoke("--iterations", "1").returncode, 0)
        self.assertEqual(self.remote_head(), self.base)
        self.set_mode("missing_report")
        self.assertEqual(self.invoke("--iterations", "1").returncode, 1)
        self.assertEqual(self.read_json(self.output / "latest.json")["status"], "blocked")
        self.assertEqual(self.remote_head(), self.base)
        self.assertFalse((self.output / "pending-push.json").exists())

    def test_failed_push_retries_same_commit_without_another_experiment(self):
        reject = self.hook(self.remote, "pre-receive", "echo 'test remote rejects push' >&2\nexit 1\n")
        failed = self.invoke("--iterations", "1")
        self.assertEqual(failed.returncode, 1, failed.stdout)
        pending = self.read_json(self.output / "pending-push.json")
        self.assertEqual(self.remote_head(), self.base)
        self.assertIn("test remote rejects push", self.read_json(self.output / "latest.json")["error"])
        reject.unlink()
        retried = self.invoke("--iterations", "1")
        self.assertEqual(retried.returncode, 0, retried.stderr)
        self.assertEqual(self.remote_head(), pending["commit"])
        self.assertEqual((self.repo / "fake_count.txt").read_text(), "1")
        self.assertFalse((self.output / "pending-push.json").exists())

    def test_acknowledged_late_push_does_not_duplicate_a_commit(self):
        reject = self.hook(self.remote, "pre-receive", "exit 1\n")
        self.assertEqual(self.invoke("--iterations", "1").returncode, 1)
        pending = self.read_json(self.output / "pending-push.json")
        reject.unlink()
        self.git("push", "origin", f"{pending['commit']}:refs/heads/master")
        self.assertEqual(self.invoke("--iterations", "1").returncode, 0)
        self.assertEqual(self.remote_head(), pending["commit"])
        self.assertEqual((self.repo / "fake_count.txt").read_text(), "1")

    def test_failed_ci_repairs_same_branch_before_merging(self):
        (self.remote / "fail-ci-once").touch()
        completed = self.invoke("--iterations", "1")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual((self.repo / "fake_count.txt").read_text(), "2")
        branches = auto_optimize.git(self.remote, "branch", "--list", "codex/perf-*").splitlines()
        self.assertEqual(len(branches), 1)
        self.assertEqual(auto_optimize.git(self.remote, "show", "master:scene.txt"), b"optimized round 2\n")
        self.assertEqual(self.read_json(self.output / "latest.json")["publication"]["status"], "merged")

    def test_diverged_master_is_not_overwritten(self):
        reject = self.hook(self.remote, "pre-receive", "exit 1\n")
        self.assertEqual(self.invoke("--iterations", "1").returncode, 1)
        pending = self.read_json(self.output / "pending-push.json")
        reject.unlink()
        tree = self.git("rev-parse", f"{self.base}^{{tree}}").decode().strip()
        concurrent = self.git("commit-tree", tree, "-p", self.base, "-m", "concurrent update").decode().strip()
        self.git("push", "origin", f"{concurrent}:refs/heads/master")
        failed = self.invoke("--iterations", "1")
        self.assertEqual(failed.returncode, 1)
        self.assertEqual(self.remote_head(), concurrent)
        self.assertEqual(self.read_json(self.output / "pending-push.json")["commit"], pending["commit"])
        self.assertEqual(self.read_json(self.output / "latest.json")["status"], "blocked")
        self.assertEqual((self.repo / "fake_count.txt").read_text(), "1")

    def test_commit_hook_changes_require_revalidation(self):
        self.hook(self.repo / ".git", "pre-commit", "printf 'hook changed source\\n' > scene.txt\ngit add scene.txt\n")
        failed = self.invoke("--iterations", "1")
        self.assertEqual(failed.returncode, 1)
        self.assertEqual(self.remote_head(), self.base)
        state = self.read_json(self.output / "latest.json")
        self.assertEqual(state["status"], "blocked")
        self.assertIn("differs from the validated candidate", state["error"])

    def test_pending_candidate_with_new_edits_is_not_published(self):
        reject = self.hook(self.remote, "pre-receive", "exit 1\n")
        self.assertEqual(self.invoke("--iterations", "1").returncode, 1)
        reject.unlink()
        (self.output / "worktree/scene.txt").write_text("unvalidated edit\n", encoding="utf-8", newline="\n")
        self.assertEqual(self.invoke("--iterations", "1").returncode, 1)
        self.assertEqual(self.remote_head(), self.base)
        self.assertEqual(self.read_json(self.output / "latest.json")["status"], "blocked")
        self.assertEqual((self.repo / "fake_count.txt").read_text(), "1")

    def test_resume_after_commit_before_journal_update(self):
        with patch.object(auto_optimize, "publish_pending", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            self.assertEqual(auto_optimize.main(self.arguments("--iterations", "1")), 130)
        pending = self.read_json(self.output / "pending-push.json")
        workspace = self.output / "worktree"
        auto_optimize.git(workspace, "commit", "--file", pending["message_path"])
        candidate = auto_optimize_git.revision(workspace)
        self.assertNotIn("commit", pending)
        completed = self.invoke("--iterations", "1")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.remote_head(), candidate)
        self.assertEqual((self.repo / "fake_count.txt").read_text(), "1")

    def test_remote_fast_forward_is_used_for_next_experiment(self):
        self.assertEqual(self.invoke("--iterations", "1").returncode, 0)
        head = self.remote_head()
        tree = self.git("rev-parse", f"{head}^{{tree}}").decode().strip()
        concurrent = self.git("commit-tree", tree, "-p", head, "-m", "concurrent update").decode().strip()
        self.git("push", "origin", f"{concurrent}:refs/heads/master")
        completed = self.invoke("--iterations", "1")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(auto_optimize.git(self.remote, "rev-parse", "master^").decode().strip(), concurrent)

    def test_sync_checkpoints_unfinished_experiment_when_master_changes_same_file(self):
        workspace = auto_optimize_git.open_workspace(self.repo, self.output)
        scene = workspace / "scene.txt"
        scene.write_bytes(b"staged experiment\n")
        auto_optimize.git(workspace, "add", "scene.txt")
        scene.write_bytes(b"unfinished experiment\n")
        self.git("commit", "-m", "advance upstream")
        self.git("push", "origin", "master")
        auto_optimize_git.sync_workspace(workspace)
        self.assertEqual(auto_optimize_git.revision(workspace), self.remote_head())
        self.assertFalse(auto_optimize.git(workspace, "status", "--porcelain").strip())
        ref = self.git("for-each-ref", "--format=%(refname)", "refs/optimizer-recovery").decode().strip()
        self.assertEqual(self.git("show", ref + ":scene.txt"), b"unfinished experiment\n")
        self.assertEqual(self.git("show", ref + "^2:scene.txt"), b"staged experiment\n")
        self.assertIn(ref, (workspace / "target/auto-optimize/MEMORY.md").read_text(encoding="utf-8"))

    def test_second_runner_cannot_enter_or_clear_stop_and_restart_releases_lock(self):
        self.set_mode("hold")
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        running = subprocess.Popen([sys.executable, __file__, "--fixture-runner", *self.arguments()],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
        try:
            self.wait_for(self.repo / "pids.json", running)
            stopped = self.invoke("--stop")
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            duplicate = self.invoke("--iterations", "1")
            self.assertEqual(duplicate.returncode, 1)
            self.assertIn(b"already running", duplicate.stderr)
            self.assertTrue((self.output / "STOP").exists())
            cleanup = self.invoke("--cleanup")
            self.assertEqual(cleanup.returncode, 1)
            self.assertIn(b"already running", cleanup.stderr)
            (self.repo / "RELEASE").touch()
            stdout, stderr = running.communicate(timeout=15)
            self.assertEqual(running.returncode, 0, (stdout, stderr))
        finally:
            (self.repo / "RELEASE").touch()
            auto_optimize.stop_process_tree(running)
            running.communicate(timeout=15)
        self.set_mode("success")
        restarted = self.invoke("--iterations", "1")
        self.assertEqual(restarted.returncode, 0, restarted.stderr)
        self.assertFalse((self.output / "STOP").exists())
        self.assertEqual(len(self.rounds()), 2)

    @unittest.skipUnless(os.name == "nt", "Checks native Windows process handles")
    def test_interrupt_records_state_and_stops_owned_process_tree(self):
        self.set_mode("hold")
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handles = []

        def interrupt(_seconds):
            self.wait_for(self.repo / "pids.json")
            for pid in self.read_json(self.repo / "pids.json"):
                handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
                self.assertTrue(handle, ctypes.get_last_error())
                handles.append(handle)
            raise KeyboardInterrupt

        try:
            with patch.object(auto_optimize.time, "sleep", side_effect=interrupt), redirect_stdout(io.StringIO()):
                code = auto_optimize.main(self.arguments())
            self.assertEqual(code, 130)
            state = self.read_json(self.output / "latest.json")
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["exit_code"], 130)
            self.assertEqual(len(handles), 2)
            for handle in handles:
                self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
            self.set_mode("success")
            restarted = self.invoke("--iterations", "1")
            self.assertEqual(restarted.returncode, 0, restarted.stderr)
            self.assertIn('"status": "interrupted"',
                          self.read_json(self.rounds()[1] / "received.json")["prompt"])
        finally:
            for handle in handles:
                kernel.CloseHandle(handle)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fixture-runner"]:
        auto_optimize.GitHub = LocalPullRequests
        sys.exit(auto_optimize.main(sys.argv[2:]))
    unittest.main()
