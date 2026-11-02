from __future__ import annotations

import contextvars
import io
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import structlog

from tinyworld_pipeline.observability import events
from tinyworld_pipeline.observability.logging import (
    THIRD_PARTY_LOGGERS,
    LoggingHandle,
    bind_context,
    bind_model,
    bind_run,
    bind_stage,
    configure_logging,
    get_logging_handle,
)

from .conftest import RUN_ID, SECRET


def _lines(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _configure(tmp_path: Path, **overrides: Any) -> tuple[LoggingHandle, io.StringIO, Path]:
    console = io.StringIO()
    run_log = tmp_path / "runs" / RUN_ID / "run.log"
    kwargs: dict[str, Any] = {
        "level": "INFO",
        "json_console": False,
        "no_color": False,
        "run_log_path": run_log,
        "secrets": [SECRET],
        "console_stream": console,
    }
    kwargs.update(overrides)
    return configure_logging(**kwargs), console, run_log


# ------------------------------------------------------------------------------------ dual sinks
def test_console_filters_by_level_while_run_log_keeps_debug(tmp_path: Path) -> None:
    handle, console, run_log = _configure(tmp_path)
    log = structlog.get_logger("test")
    log.debug("tripo.poll", status="running")
    log.info(events.STAGE_START)
    log.warning("rfc.result", warnings=2)
    handle.close()

    text = console.getvalue()
    assert "tripo.poll" not in text
    assert "stage.start" in text
    assert "rfc.result" in text

    records = _lines(run_log)
    assert [r["event"] for r in records] == ["tripo.poll", "stage.start", "rfc.result"]
    assert [r["level"] for r in records] == ["debug", "info", "warning"]
    assert records[0]["status"] == "running"
    assert records[2]["warnings"] == 2


def test_every_run_log_line_has_timestamp_level_event(tmp_path: Path) -> None:
    handle, _console, run_log = _configure(tmp_path, level="DEBUG")
    log = structlog.get_logger("test")
    log.debug("a.b")
    log.info("c.d", k=1)
    log.error("e.f")
    handle.close()

    for record in _lines(run_log):
        assert record["timestamp"].endswith("Z")
        assert record["level"] in {"debug", "info", "error"}
        assert "." in record["event"]
        assert record["logger"] == "test"


def test_secret_redacted_in_both_sinks_including_exception(tmp_path: Path) -> None:
    handle, console, run_log = _configure(tmp_path)
    log = structlog.get_logger("test")
    log.info("tripo.request", headers={"Authorization": f"Bearer {SECRET}"}, url=f"?k={SECRET}")
    try:
        raise RuntimeError(f"denied for {SECRET}")
    except RuntimeError:
        log.exception(events.STAGE_FAIL)
    handle.close()

    assert SECRET not in console.getvalue()
    assert "****wxyz" in console.getvalue()
    raw = run_log.read_text(encoding="utf-8")
    assert SECRET not in raw
    fail = _lines(run_log)[1]
    assert fail["event"] == "stage.fail"
    assert "RuntimeError: denied for ****wxyz" in fail["exception"]


def test_add_secret_applies_to_subsequent_lines(tmp_path: Path) -> None:
    handle, _console, run_log = _configure(tmp_path, secrets=())
    log = structlog.get_logger("test")
    log.info("one", key=SECRET)
    handle.add_secret(SECRET)
    log.info("two", key=SECRET)
    handle.close()

    first, second = _lines(run_log)
    assert first["key"] == SECRET
    assert second["key"] == "****wxyz"
    assert handle.secrets == (SECRET,)


def test_no_run_log_when_path_is_none(tmp_path: Path) -> None:
    handle, console, _ = _configure(tmp_path, run_log_path=None)
    structlog.get_logger("test").info("stage.start")
    assert handle.run_log_path is None
    assert len(handle.handlers) == 1
    assert "stage.start" in console.getvalue()
    assert not list((tmp_path / "runs").glob("**/*.log")) if (tmp_path / "runs").exists() else True


# ------------------------------------------------------------------------------------ bound keys
def test_bound_keys_present_on_every_line_across_threads(tmp_path: Path) -> None:
    handle, _console, run_log = _configure(tmp_path, level="DEBUG")
    log = structlog.get_logger("run")
    plan = {"hero": ("prepare", "texture"), "crate": ("prepare",)}

    def worker(model_id: str) -> None:
        with bind_model(model_id):
            for stage in plan[model_id]:
                with bind_stage(stage):
                    log.info(events.STAGE_START)
                    with bind_context(job_id=f"{model_id}-{stage}-job", attempt=1):
                        log.debug("blender.spawn", exe="blender")
                    log.info(events.STAGE_DONE, duration_s=0.1)

    with bind_run(RUN_ID):
        log.info(events.RUN_START, models=sorted(plan))
        threads = [
            threading.Thread(target=contextvars.copy_context().run, args=(worker, model_id))
            for model_id in plan
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log.info(events.RUN_DONE)
    handle.close()

    records = _lines(run_log)
    assert len(records) == 2 + 3 * 3  # run.start/done + 3 stage runs x 3 lines
    assert all(r["run_id"] == RUN_ID for r in records)
    stage_lines = [r for r in records if r["event"].startswith(("stage.", "blender."))]
    assert all(r["model_id"] in plan and r["stage"] in plan[r["model_id"]] for r in stage_lines)
    spawns = [r for r in records if r["event"] == "blender.spawn"]
    assert {(r["model_id"], r["stage"], r["job_id"], r["attempt"]) for r in spawns} == {
        ("hero", "prepare", "hero-prepare-job", 1),
        ("hero", "texture", "hero-texture-job", 1),
        ("crate", "prepare", "crate-prepare-job", 1),
    }
    # job_id does not leak past its block, model_id/stage not past theirs
    done = next(r for r in records if r["event"] == "run.done")
    assert "model_id" not in done
    assert "stage" not in done
    assert all("job_id" not in r for r in records if r["event"] == "stage.done")


# ------------------------------------------------------------------------------------ tee
def test_tee_adds_and_removes_handler_and_captures_only_inside(tmp_path: Path) -> None:
    handle, _console, run_log = _configure(tmp_path)
    log = structlog.get_logger("test")
    tee_path = tmp_path / "hero" / ".pipeline" / "logs" / f"prepare-{RUN_ID}.log"
    before = len(handle.handlers)

    log.info("outside.before")
    with handle.tee(tee_path) as yielded:
        assert yielded == tee_path
        assert len(handle.handlers) == before + 1
        log.debug("inside.debug")
        log.info("inside.info")
    log.info("outside.after")
    handle.close()

    assert not handle.handlers
    assert [r["event"] for r in _lines(tee_path)] == ["inside.debug", "inside.info"]
    assert [r["event"] for r in _lines(run_log)] == [
        "outside.before",
        "inside.debug",
        "inside.info",
        "outside.after",
    ]


def test_tee_match_filters_other_models(tmp_path: Path) -> None:
    handle, _console, _ = _configure(tmp_path)
    log = structlog.get_logger("test")
    tee_path = tmp_path / "hero.log"
    with handle.tee(tee_path, match={"model_id": "hero"}):
        with bind_model("hero"):
            log.info("hero.line")
        with bind_model("crate"):
            log.info("crate.line")
        logging.getLogger("httpx").warning("foreign line")
    handle.close()
    assert [r["event"] for r in _lines(tee_path)] == ["hero.line"]


def test_tee_handler_removed_on_exception(tmp_path: Path) -> None:
    handle, _console, _ = _configure(tmp_path)
    before = len(handle.handlers)
    with pytest.raises(RuntimeError), handle.tee(tmp_path / "x.log"):
        raise RuntimeError("boom")
    assert len(handle.handlers) == before


def test_close_closes_open_tees(tmp_path: Path) -> None:
    handle, _console, _ = _configure(tmp_path)
    cm = handle.tee(tmp_path / "x.log")
    cm.__enter__()
    installed = handle.handlers
    assert len(installed) == 3
    handle.close()
    handle.close()  # idempotent
    assert not handle.handlers
    assert get_logging_handle() is None


# ------------------------------------------------------------------------------------ reconfigure
def test_reconfigure_replaces_handlers_without_duplicates(tmp_path: Path) -> None:
    root = logging.getLogger()
    baseline = len(root.handlers)
    first, _c1, _ = _configure(tmp_path)
    assert len(root.handlers) == baseline + 2
    second, console2, run_log2 = _configure(tmp_path / "second", level="DEBUG")
    assert len(root.handlers) == baseline + 2
    assert not first.handlers
    assert get_logging_handle() is second

    structlog.get_logger("test").debug("after.reconfigure")
    second.close()
    assert len(root.handlers) == baseline
    assert "after.reconfigure" in console2.getvalue()
    assert [r["event"] for r in _lines(run_log2)] == ["after.reconfigure"]
    assert root.level == logging.getLogger().level


def test_invalid_level_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        _configure(tmp_path, level="LOUD")


def test_level_is_case_insensitive(tmp_path: Path) -> None:
    handle, console, _ = _configure(tmp_path, level="warning")
    structlog.get_logger("test").info("quiet.info")
    structlog.get_logger("test").warning("loud.warning")
    assert handle.level == "WARNING"
    assert "quiet.info" not in console.getvalue()
    assert "loud.warning" in console.getvalue()


# ------------------------------------------------------------------------------------ console modes
def test_json_console_emits_json_only(tmp_path: Path) -> None:
    handle, console, _ = _configure(tmp_path, json_console=True, run_log_path=None)
    log = structlog.get_logger("test")
    log.info("stage.start", n=1)
    log.warning("rfc.result", warnings=["a"])
    handle.close()

    lines = [json.loads(line) for line in console.getvalue().splitlines()]
    assert [line["event"] for line in lines] == ["stage.start", "rfc.result"]
    assert lines[0]["n"] == 1
    assert lines[1]["warnings"] == ["a"]
    assert "\x1b[" not in console.getvalue()


class _TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_console_colors_only_on_tty_and_not_with_no_color(tmp_path: Path) -> None:
    tty = _TtyStringIO()
    handle, _, _ = _configure(tmp_path, run_log_path=None, console_stream=tty)
    structlog.get_logger("test").info("colored.line")
    handle.close()
    assert "\x1b[" in tty.getvalue()

    tty_no_color = _TtyStringIO()
    handle, _, _ = _configure(
        tmp_path, run_log_path=None, console_stream=tty_no_color, no_color=True
    )
    structlog.get_logger("test").info("plain.line")
    handle.close()
    assert "\x1b[" not in tty_no_color.getvalue()
    assert "plain.line" in tty_no_color.getvalue()

    pipe = io.StringIO()
    handle, _, _ = _configure(tmp_path, run_log_path=None, console_stream=pipe)
    structlog.get_logger("test").info("piped.line")
    handle.close()
    assert "\x1b[" not in pipe.getvalue()


class _BrokenTty(io.StringIO):
    def isatty(self) -> bool:
        raise ValueError("closed")


def test_broken_isatty_falls_back_to_no_color(tmp_path: Path) -> None:
    stream = _BrokenTty()
    handle, _, _ = _configure(tmp_path, run_log_path=None, console_stream=stream)
    structlog.get_logger("test").info("line")
    handle.close()
    assert "\x1b[" not in stream.getvalue()


def test_console_defaults_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    handle = configure_logging(
        level="INFO", json_console=True, no_color=True, run_log_path=None, secrets=()
    )
    structlog.get_logger("test").info("to.stderr")
    handle.close()
    assert "to.stderr" in err.getvalue()
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------------------------ third party
def test_third_party_loggers_set_to_warning(tmp_path: Path) -> None:
    for name in THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)
    handle, console, run_log = _configure(tmp_path, level="DEBUG")
    for name in THIRD_PARTY_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING
    logging.getLogger("httpx").info("HTTP Request: GET https://x")
    logging.getLogger("httpx").warning("retrying %s", "GET")
    handle.close()

    records = _lines(run_log)
    assert [r["event"] for r in records] == ["retrying GET"]
    assert records[0]["logger"] == "httpx"
    assert records[0]["level"] == "warning"
    assert "HTTP Request" not in console.getvalue()
