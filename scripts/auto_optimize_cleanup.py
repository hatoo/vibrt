"""Retire reproducible files from old optimizer rounds, preserving their evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat

from auto_optimize_git import save_json


RUN_NAME = re.compile(r"\d{8}T\d{6}\.\d{6}Z\Z")
FINISHED = {"accepted", "rejected", "no_change", "blocked", "failed", "interrupted"}
CACHE_DIRS = {"build", "cargo-target", "cargo_target", "build-cache", "build_cache", "cache", "temp", "tmp"}
BINARY_SUFFIXES = {".dll", ".pyd", ".pdb", ".lib", ".obj", ".o", ".a", ".so", ".dylib",
                   ".exe", ".rlib", ".rmeta", ".ptx", ".cubin", ".fatbin", ".pyc"}
ARRAY_SUFFIXES = {".npy", ".npz", ".raw"}


class UnsafeCleanupPath(ValueError):
    pass


def linked(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def checked_path(path, repo):
    """Check both the resolved boundary and every Windows reparse-point component."""
    path = Path(path).absolute()
    repo = Path(repo).resolve()
    if path == repo or not path.is_relative_to(repo) or not path.resolve().is_relative_to(repo):
        raise UnsafeCleanupPath(f"Cleanup path escapes the repository: {path}")
    current = repo
    for part in path.relative_to(repo).parts:
        current /= part
        if linked(current.lstat()):
            raise UnsafeCleanupPath(f"Cleanup does not follow links or junctions: {current}")
    return path


def read_control(path):
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid cleanup protection metadata: {path}")
    return value


def save_manifest(path, value, repo):
    checked_path(path.parent, repo)
    for candidate in (path, path.with_suffix(path.suffix + ".tmp")):
        if os.path.lexists(candidate):
            checked_path(candidate, repo)
    save_json(path, value)


def finished_rounds(repo, output):
    rounds = []
    roots = (output / "runs", output / "worktree/target/auto-optimize/runs")
    for root in roots:
        if not root.exists():
            continue
        checked_path(root, repo)
        for directory in root.iterdir():
            if not RUN_NAME.fullmatch(directory.name):
                continue
            try:
                checked_path(directory, repo)
                state_path = checked_path(directory / "state.json", repo)
                state = read_control(state_path)
                if not all(isinstance(state.get(key), str)
                           for key in ("run_id", "status", "directory", "finished_at")):
                    continue
                if (state.get("run_id") == directory.name
                        and state.get("status") in FINISHED and state.get("finished_at")
                        and Path(state.get("directory", "")).resolve() == directory.resolve()):
                    rounds.append((directory, state))
            except (OSError, ValueError):
                # Incomplete, unrecognized or linked folders are not ours to remove.
                continue
    return sorted(rounds, key=lambda item: (item[0].name, str(item[0])))


def select_payloads(repo, directory):
    files, directories, skipped = [], [], []
    stack = [directory]
    while stack:
        parent = stack.pop()
        checked_path(parent, repo)
        with os.scandir(parent) as entries:
            for entry in entries:
                path = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                if linked(info):
                    skipped.append(str(path))
                    continue
                parts = path.relative_to(directory).parts
                cache = parts[0].lower() in CACHE_DIRS
                if not cache and any(part.lower() in {"source", "sources", ".git"} for part in parts):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                    if cache or entry.name == "__pycache__":
                        directories.append(path)
                elif entry.is_file(follow_symlinks=False):
                    suffix = path.suffix.lower()
                    if cache or suffix in BINARY_SUFFIXES | ARRAY_SUFFIXES:
                        files.append((path, info.st_size))
    return files, directories, skipped


def cleanup_artifacts(repo, keep_runs=3, dry_run=False):
    """Caller must hold the supervisor lock; only explicitly managed runs qualify."""
    repo = Path(repo).resolve()
    output = repo / "target/auto-optimize"
    checked_path(output, repo)
    if keep_runs < 1:
        raise ValueError("keep_runs must be at least 1")
    # Parse both before deleting anything. Corrupt protection metadata fails closed.
    for name in ("latest.json", "pending-push.json"):
        if os.path.lexists(output / name):
            checked_path(output / name, repo)
    latest = read_control(output / "latest.json")
    pending = read_control(output / "pending-push.json")
    protected = set()
    for record, key in ((latest, "directory"), (pending, "source_run")):
        if record:
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Missing {key} in cleanup protection metadata")
            path = Path(record[key])
            if (not path.is_absolute() or not RUN_NAME.fullmatch(path.name)
                    or path.parent.resolve() not in {
                        (output / "runs").resolve(),
                        (output / "worktree/target/auto-optimize/runs").resolve()}):
                raise ValueError(f"Unknown protected round in cleanup metadata: {path}")
            protected.add(path.resolve())
    rounds = finished_rounds(repo, output)
    protected.update(path.resolve() for path, _ in rounds[-keep_runs:])
    accepted = [path for path, state in rounds if state["status"] == "accepted"]
    if accepted:
        protected.add(accepted[-1].resolve())
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), dry_run=dry_run,
                  keep_runs=keep_runs, removed_bytes=0, removed_files=0, planned_bytes=0,
                  protected=[], rounds=[], skipped_links=[], errors=[],
                  free_bytes_before=shutil.disk_usage(repo).free)
    for directory, _ in rounds:
        if directory.resolve() in protected or (directory / "KEEP_ARTIFACTS").exists():
            report["protected"].append(str(directory))
            continue
        files, directories, skipped = select_payloads(repo, directory)
        report["skipped_links"].extend(skipped)
        if not files:
            continue
        planned = sum(size for _, size in files)
        report["planned_bytes"] += planned
        row = dict(directory=str(directory), planned_bytes=planned, removed_bytes=0,
                   removed_files=0, errors=[])
        report["rounds"].append(row)
        if dry_run:
            continue
        removed = []
        try:
            for path, _ in files:
                try:
                    checked_path(directory, repo)
                    checked_path(path, repo)
                    if not path.resolve().is_relative_to(directory.resolve()):
                        raise UnsafeCleanupPath(f"File escaped its round: {path}")
                    info = path.lstat()
                    if not stat.S_ISREG(info.st_mode):
                        raise UnsafeCleanupPath(f"Artifact changed type: {path}")
                    path.unlink()
                    row["removed_bytes"] += info.st_size
                    row["removed_files"] += 1
                    removed.append(str(path.relative_to(directory)))
                except (OSError, ValueError) as exc:
                    row["errors"].append(str(exc))
            # Remove only empty cache directories; never recursively delete a tree.
            for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
                try:
                    checked_path(path, repo)
                    path.rmdir()
                except OSError:
                    pass
        finally:
            report["removed_bytes"] += row["removed_bytes"]
            report["removed_files"] += row["removed_files"]
            report["errors"].extend(row["errors"])
            if removed:
                manifest = directory / "cleanup.json"
                if manifest.exists():
                    checked_path(manifest, repo)
                save_manifest(manifest, dict(cleaned_at=report["checked_at"],
                                            removed_bytes=row["removed_bytes"], files=removed,
                                            retained="Source snapshots, patches, reports, metrics and preview images"), repo)
    report["free_bytes_after"] = shutil.disk_usage(repo).free
    if not dry_run:
        save_manifest(output / "cleanup.json", report, repo)
    return report
