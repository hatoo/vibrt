"""Publish validated optimizer work without including the user's checkout edits."""
import json
import os
from pathlib import Path
import subprocess


BRANCH = "codex/auto-optimize"
REMOTE = "refs/remotes/origin/master"


class PublishBlocked(RuntimeError):
    pass


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def git(repo, *args):
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GCM_INTERACTIVE"] = "Never"
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=environment, timeout=60).stdout


def revision(repo, ref="HEAD"):
    return git(repo, "rev-parse", "--verify", ref).decode().strip()


def ancestor(repo, older, newer):
    try:
        git(repo, "merge-base", "--is-ancestor", older, newer)
        return True
    except subprocess.CalledProcessError as exc:
        if exc.returncode != 1:
            raise
        return False


def require_branch(workspace):
    if git(workspace, "symbolic-ref", "--quiet", "HEAD").decode().strip() != f"refs/heads/{BRANCH}":
        raise PublishBlocked(f"Optimizer workspace must stay on {BRANCH}.")


def open_workspace(repo, output):
    git(repo, "fetch", "--no-tags", "origin", f"refs/heads/master:{REMOTE}")
    workspace = output / "worktree"
    if not workspace.exists():
        if git(repo, "branch", "--list", BRANCH).strip():
            git(repo, "worktree", "add", str(workspace), BRANCH)
        else:
            git(repo, "worktree", "add", "-b", BRANCH, str(workspace), REMOTE)
    actual_root = Path(git(workspace, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    common = (workspace / git(workspace, "rev-parse", "--git-common-dir").decode().strip()).resolve()
    expected = (repo / git(repo, "rev-parse", "--git-common-dir").decode().strip()).resolve()
    if actual_root != workspace.resolve() or common != expected:
        raise PublishBlocked("The optimizer worktree belongs to a different checkout.")
    require_branch(workspace)
    return workspace


def sync_workspace(workspace):
    head = revision(workspace)
    remote = revision(workspace, REMOTE)
    if head == remote:
        return
    if ancestor(workspace, head, remote) and not git(workspace, "status", "--porcelain").strip():
        git(workspace, "merge", "--ff-only", REMOTE)
        return
    raise PublishBlocked("master advanced with unfinished work, or the optimizer branch diverged; "
                         "preserving it for reconciliation and revalidation. No force push is used.")


def queue_publication(workspace, output, directory, result, baseline, previous=None):
    require_branch(workspace)
    if revision(workspace) != baseline:
        raise PublishBlocked("HEAD changed during the experiment; the agent must not create commits.")
    subject = result["commit_message"].strip()
    if not subject or "\n" in subject or "\r" in subject:
        raise PublishBlocked("An accepted improvement needs a one-line commit_message.")
    if not all(result[key] for key in ("changes", "measurements", "validation")):
        raise PublishBlocked("An accepted improvement needs changes, measurements and validation.")
    report = directory / "REPORT.md"
    if not report.is_file() or not report.read_text(encoding="utf-8").strip():
        raise PublishBlocked("An accepted improvement needs a nonempty REPORT.md.")
    git(workspace, "diff", "--check")
    git(workspace, "add", "--all", "--", ".")
    git(workspace, "diff", "--cached", "--check")
    tree = git(workspace, "write-tree").decode().strip()
    if tree == revision(workspace, "HEAD^{tree}"):
        raise PublishBlocked("Accepted result has no source changes to commit.")
    message = subject + "\n\n" + result["summary"] + "\n\n"
    for title, key in (("Measurements", "measurements"), ("Validation", "validation")):
        message += title + ":\n" + "".join(f"- {line}\n" for line in result[key]) + "\n"
    message_path = directory / "commit-message.txt"
    message_path.write_text(message, encoding="utf-8")
    pending = dict(parent=baseline, tree=tree, message_path=str(message_path),
                   source_run=str(directory), result=result)
    if previous:
        pending.update(source_run=previous["source_run"], branch=previous["branch"],
                       result=previous["result"], pr=previous["pr"])
    # Persist before committing: interruption immediately after git commit is recoverable.
    save_json(output / "pending-push.json", pending)
    return pending


def commit_pending(workspace, output, pending):
    require_branch(workspace)
    head = revision(workspace)
    if head == pending["parent"]:
        if (git(workspace, "diff", "--name-only").strip()
                or git(workspace, "ls-files", "--others", "--exclude-standard").strip()
                or git(workspace, "write-tree").decode().strip() != pending["tree"]):
            raise PublishBlocked("The validated files changed before commit; revalidation is required.")
        git(workspace, "commit", "--file", pending["message_path"])
        head = revision(workspace)
    if (git(workspace, "rev-list", "--parents", "-n", "1", head).decode().split()
            != [head, pending["parent"]]
            or revision(workspace, f"{head}^{{tree}}") != pending["tree"]
            or git(workspace, "status", "--porcelain").strip()):
        raise PublishBlocked("The committed tree differs from the validated candidate; preserving it without pushing.")
    pending["commit"] = head
    save_json(output / "pending-push.json", pending)
    return head
