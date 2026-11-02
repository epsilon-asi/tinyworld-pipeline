# 0001 — Two Python runtimes joined by a file-based job protocol

Date: 2026-11-02 · Status: accepted

## Context

Blender ships its own Python interpreter (3.11 in the 4.x series) without our dependencies. The
orchestrator needs pydantic, httpx, numpy, Pillow and friends. Two options exist for driving Blender:
the `bpy` wheel inside the project venv, or a headless Blender subprocess.

## Decision

The orchestrator runs on its own Python 3.12 venv. All Blender work runs in a fresh
`blender -b --factory-startup --python entry.py -- --job job.json --result result.json` process. The
Blender-side package `tinyworld_pipeline.blender_jobs` imports only the standard library, `bpy`,
`bmesh`, `mathutils` and Blender's bundled `numpy`. Requests and results are JSON documents; the
orchestrator validates them with pydantic, the Blender side mirrors them with frozen dataclasses. A
parity test keeps both sides' field names identical.

## Consequences

- Positive: any installed Blender 4.5 LTS works, no ABI coupling, no Blender-version pin in the venv,
  every job is replayable from its `job.json`, crashes are isolated to one process.
- Negative: per-job Blender start-up cost (a few seconds), two serialisation layers to keep in sync,
  no shared in-memory state between jobs (the `.blend` file is the state).
- Neutral: Blender-side logic is tested against a `SceneApi` fake; only the thin `BpyScene` adapter
  needs real Blender.

## Alternatives considered

- `bpy` wheel in the venv: pins the whole project to Blender's Python ABI and a single Blender
  version; wheels lag releases and are large.
- Long-running Blender server process: faster per job but adds protocol complexity and makes crash
  isolation and cancellation harder.
