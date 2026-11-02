# 0002 — macOS is the primary host; Windows host support deferred

Date: 2026-11-02 · Status: accepted

## Context

Plan 1.0 targeted Windows 11 and macOS as `gp` hosts with a Linux + Windows CI matrix. The team
builds on an Apple Silicon Mac. Unity lives on a separate Windows PC. The pipeline's deliverable per
model is a set of plain files (FBX, PNG, companion FBX clips, `manifest.json`) that Unity imports
identically on any OS.

## Decision

- macOS (Apple Silicon) is the primary and only supported host for running `gp` in v1.
- Linux is supported for CI (lint, type, unit and Blender integration jobs).
- Windows as a host for `gp` is deferred. The code stays portable: `pathlib` everywhere, forward-slash
  relative paths in JSON, argv lists (no shell strings), platform branches isolated to three places:
  `blender/locate.py` (well-known install paths), `blender/runner.py` (process-group creation and
  tree kill) and `tests/fakes/fake_blender_exe.py` (launcher wrapper).
- Unity consumption happens on the PC by copying `output/<model>/` into the project. `gp unity sync`
  remains available for a shared or synced folder but is off by default.

## Consequences

- Positive: one platform to build and maintain; Cycles bakes use Metal on the Mac; no Windows CI
  runner cost; Spike S2's Unity half becomes a file round trip to the PC.
- Negative: the Windows branches are written but unexercised until a Windows host is needed; a
  later port will need one round of CI fixes.
- Neutral: the plan's Windows-specific specifications (`07` §4, `19` §5) are retained as the
  contract for that future port.

## Alternatives considered

- Keep Windows in the CI matrix now: cost without a consumer; untestable locally.
- Drop portability habits entirely: saves nothing measurable and makes the later port expensive.
