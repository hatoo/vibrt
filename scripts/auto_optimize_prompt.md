You are running one round of a user-requested continuous optimization loop for vibrt.
The user starts this loop manually and wants it to keep improving the renderer until
they stop it. Do useful implementation work, not just a proposal. Finish this round
with the requested structured result; the supervisor will start the next round.
Do not start another loop or change the supervisor, prompt, or acceptance criteria.

Repository: {{REPO}}
Original checkout (read-only; local scenes and historical artifacts): {{SOURCE_REPO}}
This round's artifacts: {{ROUND}}
Persistent experiment notes: {{MEMORY}}

The user authorized automatic publication of verified improvements to origin/master.
You work on the dedicated codex/auto-optimize worktree, starting from remote master;
the supervisor opens a PR for each accepted improvement and merges it after CI passes. Keep the
complete tested changes in this worktree. Do not create commits, stage files, switch
branches, merge, rebase or push yourself. Do not modify the supervisor's publication
journal, or change the original checkout, global settings or installed Blender addon.
Follow current project conventions. Preserve unfinished work from earlier rounds;
do not reset, stash or clean it. Use separate headless Blender processes without
closing or modifying the user's open scene.

## Check artifact access first

Before reading historical reports or planning an experiment, create a small probe
file and a nested directory inside this round's artifact directory, then read the
file back. All new reports, scripts, binaries, renders and persistent notes must
stay at the supplied paths inside this worktree. They do not require writes to the
original checkout. If this initial check fails, immediately return blocked with
the exact path and error; do not spend the round repeating the historical analysis.
Do not modify filesystem ACLs, sandbox settings or approval policy to resolve it.
The worktree's target/auto-optimize contains live logs and reports: do not delete
the whole target directory or run an unscoped cargo clean. Clean an affected package
only when a rebuild requires it.

The user authorized periodic artifact cleanup by the supervisor between rounds.
It retains recent rounds, the latest accepted improvement and pending publication;
old build caches, native binaries and NumPy/raw buffers may be removed. Source
snapshots, patches, reports, measurement JSON and preview images are kept. Check a
prior run's cleanup.json before treating a missing generated file as a new failure.
Rebuild required artifacts from the correct preserved source and verify their identity.
If an unfinished experiment needs older binary/array artifacts beyond the normal
retention window, create KEEP_ARTIFACTS in that run and explain why in MEMORY.md;
remove your marker when the dependency is no longer needed. Do not clean other runs
yourself. Complete all build/render jobs before returning so cleanup can run safely.

## Start with current evidence

Read the current Git state, previous round state/result, and persistent experiment
notes. If the last round was interrupted, failed, or left marked running after a
supervisor exit, recover its unfinished experiment before starting another; do not
assume its edits are validated. The supervisor's
before.patch includes pre-existing tracked changes and is evidence, not a patch to
blindly reverse. Preserve pre-existing untracked files too.
Older rounds may have used the original checkout. Their edits are not automatically
present here: inspect the relevant optimization hunks, port useful ones into this
worktree, and validate against this worktree's actual baseline before accepting them.

Inspect current renderer/exporter code and choose one concrete performance hypothesis.
Focus on Cozy Kitchen, then Junk Shop and an affected additional scene. Track export,
compilation/preparation, time to first preview and sampling separately. Use profiling
or small controlled experiments to choose the next change. Avoid repeating rejected
ideas unless new evidence changes the hypothesis.

Historical context only: the 2026-11-22 validated build removed late material
optixReorder and measured Cozy Kitchen at 23.93 s / 1024 samples and Junk Shop at
9.53 s / 256 samples on an RTX 5080, 768 x 723, with denoising disabled. These are
not a current baseline: subsequent commits changed path roulette. Rebuild and measure
the actual starting working tree. Historical reports, if still available, live under
{{SOURCE_REPO}}/target/kitchen-next and {{SOURCE_REPO}}/target/kitchen-opt. Immediate Noise/Mapping operand packing
did not demonstrate a meaningful improvement; the initial no-reorder trial had stale
build artifacts and missing procedural detail, and was rejected by image comparison.

## Implement and measure one experiment

1. Save this round's source baseline, including any files you will change, without
   losing earlier work. Build a baseline extension before editing. Save its SHA-256
   and source identity under this round's directory. Preserve the known good binary.
2. Implement the selected optimization. Build the candidate and record its source
   identity and binary SHA-256. When restoring files, make sure Cargo/C++ actually
   rebuild affected dependencies; old copied modification times previously caused
   incompatible code to be linked. Do not trust a fast render without checking images.
3. Compare baseline and candidate sequentially on the same GPU, scene, camera,
   resolution, seed, samples, bounce settings, clamps, denoising and preview mode.
   Warm up each build, then perform at least two comparable measured runs. For the
   main Cozy Kitchen check use 768 x 723 at 1024 samples; use 256 samples for a quick
   screen and for Junk Shop. Keep export and shader compilation out of sampling time.
   Retain full shader graphs, SSS, lighting and sample counts. Diagnostic feature
   removals are not acceptable performance improvements.
4. Invoke native rendering inside Blender's actual RenderEngine callback so borrowed
   RNA/mesh buffers remain alive throughout the measurement. The historical
   {{SOURCE_REPO}}/target/kitchen-next/bench.py demonstrates this. Do not hold borrowed pointers after
   their depsgraph lifetime. Create a durable harness in scripts/ if needed; preserve
   all measurement conditions and save linear pixels plus useful preview images.
   Load addon/exporter code and native binaries from this worktree, not accidentally
   from the original checkout or the installed addon.
5. Compare linear RGB buffers, finite values, image structure, brightness and noise.
   Inspect renders visually. For algorithmic sampling changes, also compare time to
   equal quality against a sufficiently converged reference; equal-sample speed alone
   cannot justify worse convergence. Do not adopt changes that lose materials,
   illumination, energy, shader features or correctness.
6. Run checks relevant to the change: cargo fmt --all -- --check, git diff --check,
   Rust tests, Python exporter tests, and scripts/check_struct_layouts.py when applicable.
   Use existing GPU fixtures such as test_shader_nodes.py, test_shader_graphs.py,
   test_scene_nodes.py, test_scene_parity.py and test_bsdf_mixtures.py for affected paths.
   Compare the tracked IES scene and an affected real scene for rendering changes.
   Stop widening tests after appropriate checks pass unless new evidence warrants it.
7. Retain only a repeatable improvement larger than ordinary timing variation, with
   acceptable images and passing checks. If it regresses, fails, or has no clear benefit,
   revert only this round's edits and leave a concrete next hypothesis. Do not report
   success from a stale binary or a failed/incomplete benchmark.

Run GPU work serially. If another render occupies the GPU, do useful CPU analysis
or return blocked with evidence instead of collecting misleading timings. Do not kill
unrelated processes. Wait for every build/render job you start before ending the
round, or stop your own unfinished jobs. Authentication, missing dependencies and unavailable hardware
must be reported honestly; do not weaken execution restrictions to bypass them.

## Local tools and continuity

The usual Blender executable is C:/Program Files/Blender Foundation/Blender 5.1/blender.exe;
verify availability. The native build command is cargo build --release -p vibrt --features python.
PYO3_PYTHON is provided by the supervisor when not already configured. The relevant scenes are:

- {{SOURCE_REPO}}/test_scenes/cozy_kitchen/blender-3.5-splash.blend
- {{SOURCE_REPO}}/test_scenes/junk_shop/junk_shop/junk_shop.blend
- test_scenes/ies_light/ies_light/ies_light.blend (tracked in this worktree)

Store measurements, source/binary identities, logs and comparison renders under this
round's directory. Write a concise REPORT.md with the problem, retained/rejected change,
commands, repeated timings, image differences, test results and limitations. Update
the persistent notes with accepted/rejected hypotheses and the next useful experiment,
so a fresh Codex process can continue without redoing completed work. Reference older
reports instead of copying the entire history into the notes.

Return the requested JSON result with a Japanese summary and a concrete next_step.
Use accepted only for a verified improvement, rejected for a reverted experiment,
no_change for useful investigation without a code change, and blocked for a concrete
external impediment. Do not claim a report or measurement that was not produced.
An accepted result must include nonempty changes, measurements and validation lists,
a complete REPORT.md, and a concise one-line commit_message such as
"renderer: reuse prepared pipelines between renders". Use an empty commit_message
for other statuses. Include relevant commands and measured results so the supervisor
can put the evidence in the commit body. All changes in this dedicated worktree will
be included, so remove rejected experiment edits and keep artifacts in the specified
round directory or ignored build directories. The supervisor resumes the same PR after interruption and repairs failed CI before merging.
