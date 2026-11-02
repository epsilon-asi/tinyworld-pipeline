"""Production assembly for the CLI: stage graph, adapters, library root, Blender, secrets.

Nothing here prints. The CLI commands call these helpers; tests replace the pieces that touch the
outside world through :class:`~grounded_pipeline.cli.state.CliHooks`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import structlog
from pydantic import SecretStr

from grounded_pipeline import __version__
from grounded_pipeline.blender.locate import (
    BlenderInfo,
    blender_version,
    check_version,
    locate_blender,
)
from grounded_pipeline.blender.protocol import JobRequest, JobResult
from grounded_pipeline.blender.runner import (
    BlenderRunner,
    SubprocessBlenderRunner,
    default_entry_script,
)
from grounded_pipeline.config.layering import CliOverrides
from grounded_pipeline.config.settings import Secrets, load_secrets
from grounded_pipeline.domain.errors import (
    BlenderNotFoundError,
    BlenderVersionError,
    PipelineError,
    Problem,
    TripoAuthError,
    UsageError,
)
from grounded_pipeline.domain.library_config import BlenderSection, LibraryConfig, TripoSection
from grounded_pipeline.domain.ports import CancelToken
from grounded_pipeline.domain.state import StageName, StageOutcome
from grounded_pipeline.library.discovery import Library
from grounded_pipeline.library.paths import LibraryPaths, find_library_root
from grounded_pipeline.library.validation import BlenderCheck
from grounded_pipeline.observability.logging import LoggingHandle
from grounded_pipeline.pipeline.context import ModelContext, PipelineContext
from grounded_pipeline.pipeline.graph import StageGraph
from grounded_pipeline.pipeline.hashing import sha_text
from grounded_pipeline.pipeline.locks import blender_semaphore
from grounded_pipeline.pipeline.runner import ContextFactory, RunSetup
from grounded_pipeline.pipeline.stage import Stage
from grounded_pipeline.pipeline.stages.apply import ApplyStage, BakeRoute
from grounded_pipeline.pipeline.stages.discover import DiscoverStage
from grounded_pipeline.pipeline.stages.export import ExportStage
from grounded_pipeline.pipeline.stages.handoff import HandoffStage
from grounded_pipeline.pipeline.stages.prepare import PrepareStage
from grounded_pipeline.pipeline.stages.texture import TextureStage
from grounded_pipeline.pipeline.stages.verify import VerifyStage
from grounded_pipeline.pipeline.state_store import JsonStateStore
from grounded_pipeline.pipeline.timing import SystemClock, SystemSleeper
from grounded_pipeline.texturing.provider import TextureProvider, TextureRequest, TextureResult
from grounded_pipeline.texturing.tripo_provider import TripoTextureProvider
from grounded_pipeline.tripo.client import TripoClient
from grounded_pipeline.tripo.polling import RateLimiter

if TYPE_CHECKING:
    from grounded_pipeline.cli.state import CliState

ENV_PREFIX = "GP_"
W_BLENDER_VERSION = "W_BLENDER_VERSION"
BLENDER_CACHE_FILENAME = "blender.json"
RUN_LOG_FILENAME = "run.log"


# --------------------------------------------------------------------------- stage graph
class NotYetImplementedStage:
    """Placeholder for a stage whose work package has not landed yet (plan 16 wave plan).

    Register it in :data:`STAGE_FACTORIES` to keep the graph complete while a stage is pending.
    """

    def __init__(self, name: StageName) -> None:
        self.name: StageName = name

    def input_fingerprint(self, model: ModelContext, ctx: PipelineContext) -> str:
        """Constant per stage name, so plans stay stable."""
        del model, ctx
        return sha_text(f"not-implemented:{self.name}")

    def run(self, model: ModelContext, ctx: PipelineContext) -> StageOutcome:
        """Always fails.

        Raises:
            PipelineError: ``stage not implemented yet``.
        """
        del model, ctx
        raise PipelineError(
            f"stage {self.name!r} not implemented yet",
            detail={"stage": self.name},
            hint=f"run with --to-stage apply until the {self.name} stage lands",
        )


StageFactory = Callable[[BakeRoute | None], Stage]

STAGE_FACTORIES: dict[StageName, StageFactory] = {
    "discover": lambda _bake: DiscoverStage(),
    # PrepareStage declares ``name`` as a ClassVar; the Stage protocol wants an instance attribute.
    "prepare": lambda _bake: cast("Stage", PrepareStage()),
    "texture": lambda _bake: TextureStage(),
    "verify": lambda _bake: VerifyStage(),
    "apply": lambda bake: ApplyStage(bake_route=bake),
    "export": lambda _bake: cast("Stage", ExportStage()),
    "handoff": lambda _bake: cast("Stage", HandoffStage()),
}
"""Stage registry (all seven v1 stages)."""


def build_graph(bake_route: BakeRoute | None = None) -> StageGraph:
    """The v1 linear graph from :data:`STAGE_FACTORIES`."""
    return StageGraph([factory(bake_route) for factory in STAGE_FACTORIES.values()])


# --------------------------------------------------------------------------- overrides
def cli_overrides_from(
    state: CliState,
    *,
    concurrency: int | None = None,
    route: str | None = None,
    max_credits: float | None = None,
) -> CliOverrides:
    """Plan 08 §3 flag → config-key mapping."""
    return CliOverrides.model_validate(
        {
            "blender_executable": state.blender,
            "concurrency": concurrency,
            "route": route,
            "max_credits": max_credits,
            "log_level": state.log_level,
        }
    )


# --------------------------------------------------------------------------- library
def resolve_library_root(state: CliState, env: Mapping[str, str] | None = None) -> Path:
    """``--library`` > ``GP_LIBRARY`` > nearest ancestor with ``library.toml`` > ``./library``.

    Raises:
        UsageError: Nothing matched.
    """
    root = find_library_root(
        state.cwd, env if env is not None else state.env, explicit=state.library
    )
    if root is None:
        raise UsageError(
            "no asset library found",
            hint="pass --library PATH, set GP_LIBRARY, or run inside a library (gp init)",
        )
    return root


def load_library(state: CliState) -> Library:
    """Resolve the root and load the library (leniently; see ``Library.load``).

    Raises:
        UsageError: No root could be resolved or it has no ``library.toml``.
    """
    root = resolve_library_root(state)
    paths = LibraryPaths.from_root(root)
    if not paths.library_toml.is_file():
        raise UsageError(
            f"{paths.library_toml} not found",
            hint="run `gp init` there first, or point --library at an existing library",
        )
    return Library.load(root)


def gp_environment(env: Mapping[str, str]) -> dict[str, str]:
    """The ``GP_*`` subset of ``env`` (what the discover fingerprint and reports record)."""
    return {k: v for k, v in env.items() if k.startswith(ENV_PREFIX)}


# --------------------------------------------------------------------------- secrets
def secrets_from_mapping(env: Mapping[str, str]) -> Secrets:
    """Secrets from an explicit mapping only (no ``.env`` files, no process environment)."""
    key = env.get("TRIPO_API_KEY")
    return Secrets.model_construct(tripo_api_key=SecretStr(key) if key else None)


def default_secrets_loader() -> Secrets:
    """Production loader: process environment and the two ``.env`` locations (plan 07 §2)."""
    return load_secrets()


# --------------------------------------------------------------------------- blender
def resolve_blender(
    explicit: Path | None, env: Mapping[str, str], section: BlenderSection
) -> BlenderInfo:
    """Locate the executable and read its version (plan 07 §4).

    Raises:
        BlenderNotFoundError: No executable.
        BlenderVersionError: ``--version`` output unparseable.
    """
    exe = locate_blender(explicit, env, section)
    return BlenderInfo(path=exe, version=blender_version(exe))


def blender_problems(
    state: CliState, section: BlenderSection
) -> tuple[BlenderInfo | None, list[Problem]]:
    """Resolve Blender and apply the version policy as validation problems.

    Returns:
        ``(info or None, problems)`` — an error problem when Blender is missing or refused, a
        ``W_BLENDER_VERSION`` warning for tolerated versions.
    """
    try:
        info = state.hooks.blender_resolver(state.blender, state.env, section)
        warnings = check_version(info.version, allow_unsupported=section.allow_unsupported)
    except (BlenderNotFoundError, BlenderVersionError) as exc:
        return None, [
            Problem(
                code=exc.code,
                severity="error",
                path="blender",
                message=exc.message,
                hint=exc.hint,
            )
        ]
    return info, [
        Problem(code=W_BLENDER_VERSION, severity="warning", path="blender", message=w)
        for w in warnings
    ]


def blender_check(state: CliState) -> BlenderCheck:
    """``validate_library(blender_check=...)`` hook bound to the CLI state."""

    def _check(config: LibraryConfig) -> list[Problem]:
        return blender_problems(state, config.blender)[1]

    return _check


def write_blender_cache(paths: LibraryPaths, info: BlenderInfo) -> Path:
    """Record the resolved executable in ``library/.pipeline/blender.json`` (plan 07 §4)."""
    target = paths.pipeline_dir / BLENDER_CACHE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"path": str(info.path), "version": info.version_string}, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


# --------------------------------------------------------------------------- tripo
def make_tripo_client(api_key: SecretStr, section: TripoSection) -> TripoClient:
    """Production client over ``[tripo]`` settings with system clock/sleeper."""
    clock, sleeper = SystemClock(), SystemSleeper()
    limiter = RateLimiter(
        section.max_concurrent_tasks, section.min_request_interval_s, clock=clock, sleeper=sleeper
    )
    return TripoClient(
        api_key, str(section.base_url), sleeper=sleeper, clock=clock, rate_limiter=limiter
    )


class _NoBlender:
    """Blender runner used when no executable is needed/available (``--to-stage discover``)."""

    def run(
        self, req: JobRequest, *, timeout_s: int, log_path: Path, cancel: CancelToken
    ) -> JobResult:
        del req, timeout_s, log_path, cancel
        raise BlenderNotFoundError(
            "this run was started without a Blender executable",
            hint="pass --blender PATH or set [blender].executable in library.toml",
        )


class _NoProvider:
    """Texture provider used when ``TRIPO_API_KEY`` is absent (``--to-stage prepare``)."""

    def estimate_credits(self, req: TextureRequest) -> float:
        del req
        return 0.0

    def texture(
        self,
        req: TextureRequest,
        *,
        resume_task_id: str | None,
        cancel: CancelToken,
        on_task_created: Callable[[str], None] | None = None,
    ) -> TextureResult:
        del req, resume_task_id, cancel, on_task_created
        raise TripoAuthError(
            "TRIPO_API_KEY is not set", hint="export TRIPO_API_KEY or put it in .env"
        )


# --------------------------------------------------------------------------- context
def build_production_context(
    setup: RunSetup,
    *,
    state: CliState,
    secrets: Secrets,
    blender: BlenderInfo | None,
    logging_handle: LoggingHandle | None = None,
    overrides: CliOverrides | None = None,
) -> PipelineContext:
    """Assemble the real :class:`PipelineContext` for one run.

    Args:
        setup: Library, options, run id and budget decided by the runner.
        state: CLI state (environment, ``--blender``, hooks).
        secrets: Loaded secrets; a missing key yields a provider that refuses to texture.
        blender: Resolved executable; ``None`` yields a runner that refuses to run jobs.
        logging_handle: When given, ``runs/<run_id>/run.log`` (JSON lines, DEBUG) is teed from
            it for the rest of the invocation.
        overrides: CLI overrides recorded in the context (``cli_overrides``).
    """
    if logging_handle is not None:
        run_log = setup.library.paths.runs_dir / setup.run_id / RUN_LOG_FILENAME
        state.exit_stack.enter_context(logging_handle.tee(run_log))
    cfg = setup.library.config
    env = dict(state.env)
    blender_runner: BlenderRunner = _NoBlender()
    if blender is not None:
        blender_runner = SubprocessBlenderRunner(
            blender.path,
            default_entry_script(),
            blender_semaphore(cfg.library.blender_parallelism),
            env,
        )
    provider: TextureProvider = _NoProvider()
    if secrets.has_tripo_key and secrets.tripo_api_key is not None:
        client = state.hooks.tripo_client_factory(secrets.tripo_api_key, cfg.tripo)
        provider = TripoTextureProvider(
            client,
            credit_table=cfg.tripo.credit_table,
            poll_interval_s=cfg.tripo.poll_interval_s,
            poll_timeout_s=cfg.tripo.poll_timeout_s,
            presign_above_mb=cfg.tripo.use_presign_above_mb,
        )
    cancel = CancelToken()
    state.active_cancel = cancel
    return PipelineContext(
        library=setup.library,
        options=setup.options,
        run_id=setup.run_id,
        blender=blender_runner,
        provider=provider,
        state=JsonStateStore(setup.library.paths, pipeline_version=__version__),
        budget=setup.budget,
        cancel=cancel,
        clock=SystemClock(),
        sleeper=SystemSleeper(),
        log=structlog.get_logger("grounded_pipeline.run").bind(run_id=setup.run_id),
        pipeline_version=__version__,
        env=gp_environment(env),
        cli_overrides=(overrides or CliOverrides()).model_dump(mode="json"),
    )


def production_context_factory(
    state: CliState,
    *,
    secrets: Secrets,
    blender: BlenderInfo | None,
    overrides: CliOverrides,
) -> ContextFactory:
    """Bind :func:`build_production_context` to its per-invocation arguments."""

    def _factory(setup: RunSetup) -> PipelineContext:
        return build_production_context(
            setup,
            state=state,
            secrets=secrets,
            blender=blender,
            logging_handle=state.logging_handle,
            overrides=overrides,
        )

    return _factory


def process_environment() -> Mapping[str, str]:
    """The live process environment (indirection so tests can tell it apart)."""
    return os.environ


__all__ = [
    "BLENDER_CACHE_FILENAME",
    "RUN_LOG_FILENAME",
    "STAGE_FACTORIES",
    "W_BLENDER_VERSION",
    "NotYetImplementedStage",
    "blender_check",
    "blender_problems",
    "build_graph",
    "build_production_context",
    "cli_overrides_from",
    "default_secrets_loader",
    "gp_environment",
    "load_library",
    "make_tripo_client",
    "process_environment",
    "production_context_factory",
    "resolve_blender",
    "resolve_library_root",
    "secrets_from_mapping",
    "write_blender_cache",
]
