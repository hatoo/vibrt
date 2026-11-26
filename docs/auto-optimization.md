# Continuous local optimization

Start the loop manually from PowerShell:

```powershell
py C:/Users/hato2/Desktop/vibrt/scripts/auto_optimize.py
```

The default is unlimited rounds. Each round runs `codex exec`, reads the previous
result and experiment notes, implements one optimization, and compares performance
and images. Each verified improvement gets a PR targeting `master`. The configured Codex model and login are inherited. No recurring
schedule is installed.

## Automatic publication

The runner fetches `origin/master` and uses the dedicated `codex/auto-optimize`
branch under `target/auto-optimize/worktree/`. It leaves the original checkout's
branch, index and unfinished changes in place. Local scene assets are read from the
original checkout; builds and validation use the dedicated worktree's source.

Only an `accepted` result with changes, measurements, validation, a commit subject
and a nonempty report can be published. Codex assesses performance and image quality;
the supervisor pushes a unique `codex/perf-*` branch and opens or resumes its PR.
The **Renderer checks** job, including `deslop-please --features python`, must
succeed. The supervisor checks other jobs and commit statuses on the same SHA
and requests a merge commit for that SHA. GitHub enforces branch protections.

Failed checks trigger a Codex repair in the same worktree. Codex receives the
failed checks and annotations, runs affected local validation and submits a
follow-up commit to the same PR. The publication journal survives restarts.

For API access the supervisor uses Git Credential Manager for github.com or
`GH_TOKEN` / `GITHUB_TOKEN` with pull-request and content write access and
read access to checks, commit statuses and Actions. Git pushes use the configured
origin credentials. Enable merge commits in the repository.

The CLI runs with the workspace-write sandbox and no interactive approval prompts.
Dependencies and GPU access must already work in that environment. A blocked run
records the limitation and retries with increasing delays, up to 15 minutes.
Completed rounds normally have a 10-second gap. The loop does not change permission
settings when a command is denied.

## Artifact cleanup

At startup and after each completed round, the supervisor removes reproducible
payloads from old, finished rounds. It keeps the newest **3** finished rounds, the
latest round, the latest accepted improvement and any pending publication. A run
containing a `KEEP_ARTIFACTS` file is also retained in full.

Cleanup removes old run-local build/cache/temp directories, compiled binaries and
NumPy/raw buffers. Source snapshots, patches, reports, measurement JSON, logs,
scripts and preview images outside caches remain available. Per-run `cleanup.json`
files record removed paths. Older binaries and arrays may need rebuilding when
revisiting an experiment. Cleanup covers both legacy and current optimizer runs;
it does not clean the original checkout's ordinary builds or scene files.

The same repository lock prevents cleanup during an active optimizer round. Paths
are checked before deletion, links/junctions are skipped, and unknown or incomplete
run metadata is preserved. If less than **10 GiB** remains after cleanup, the loop
waits before starting another build/render. Pending commit/push retries can continue.

```powershell
py scripts/auto_optimize.py --cleanup --dry-run  # inspect eligible runs and sizes
py scripts/auto_optimize.py --cleanup            # clean now, without Codex or pushes
py scripts/auto_optimize.py --keep-runs 5 --min-free-gib 20
```

The normal launch command enables cleanup automatically. The last cleanup summary
is saved to `target/auto-optimize/cleanup.json`. Stop an older running supervisor
and restart it to load these settings.

## Stop and restart

- **Ctrl+C** interrupts the current CLI and its child processes. The next start reads
  the interrupted round's state and recovers before trying another change.
- To finish the current round before stopping, run this in another terminal:

  ```powershell
  py C:/Users/hato2/Desktop/vibrt/scripts/auto_optimize.py --stop
  ```

Start the original command again to continue. A per-repository OS file lock prevents
two loops from running together and releases automatically if the runner exits.
Avoid simultaneous edits in the dedicated worktree or competing GPU benchmark runs.
An interrupted experiment can leave unfinished changes, so read the latest result
before using them. Existing changes are preserved; tracked before/after
patches are recorded, while untracked files must be preserved by the optimizing agent.

## Inspect and configure

```powershell
py scripts/auto_optimize.py --dry-run
py scripts/auto_optimize.py --check-sandbox
py scripts/auto_optimize.py --iterations 1
py scripts/auto_optimize.py --pause-seconds 30
```

`--repo` selects another checkout. `--codex` selects a native CLI executable.
On Windows the runner prefers the npm package's native executable and falls back
to `codex.exe` on PATH. It does not execute a shell command assembled from the prompt.

`--check-sandbox` runs one small diagnostic using the configured Codex model and the
same sandbox. It checks file creation, a nested copy, and an `apply_patch` write.
It performs no optimization or push, returns nonzero when blocked, and records the
result in `target/auto-optimize/sandbox-check.json`. Run it from the same PowerShell
terminal used to launch the loop when diagnosing Windows permission failures.

Results are gitignored under `target/auto-optimize/`:

- `latest.json`: current/last round, status, summary and artifact directory.
- `pending-push.json`: validated candidate and commit awaiting publication/retry.
- `sandbox-check.json`: result of the optional real sandbox write diagnostic.
- `cleanup.json`: retention decisions, deleted byte counts, free space and errors.
- `worktree/`: isolated optimizer branch, source and build outputs.
- `worktree/target/auto-optimize/MEMORY.md`: experiment notes maintained by Codex.
- `worktree/target/auto-optimize/runs/<timestamp>/`: prompt, result, events, stderr, snapshots,
  and the round's report/render artifacts.
- `runs/<timestamp>/`: earlier runs and failures before worktree setup completes.

Agent-written artifacts live inside its primary workspace so saving them does not
depend on an extra writable directory outside the worktree. Older `MEMORY.md` notes
are imported once; old runs remain available. Each round checks artifact writes
before lengthy analysis. The supervisor keeps its lock, stop signal and publication
journal outside the agent's workspace. Do not remove the worktree's entire `target/`
directory while the loop runs, since it now contains live logs and reports.

The workflow instructions are in
[`scripts/auto_optimize_prompt.md`](../scripts/auto_optimize_prompt.md). The supervisor
repeats the process and records results; Codex performs the implementation, measurements
and acceptance decision. Each published commit includes measurements and validation
in its message. Updates to the installed Blender addon remain manual steps.

When upgrading an already running supervisor, finish its current round with `--stop`
and start it again to load the new script and publication behavior.

The supervisor can be tested against a local bare Git remote, without API calls,
GitHub pushes or GPU work:

```powershell
py scripts/test_auto_optimize.py -v
py scripts/test_auto_optimize_cleanup.py -v
```
