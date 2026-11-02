from __future__ import annotations

import re

from grounded_pipeline.observability import events

_EVENT_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


def _declared_constants() -> dict[str, str]:
    return {
        name: value
        for name, value in vars(events).items()
        if name.isupper() and isinstance(value, str) and name != "ALL_EVENTS"
    }


def test_every_event_is_noun_dot_verb() -> None:
    for event in events.ALL_EVENTS:
        assert _EVENT_RE.match(event), event


def test_events_are_unique() -> None:
    assert len(set(events.ALL_EVENTS)) == len(events.ALL_EVENTS)


def test_all_events_lists_every_constant() -> None:
    assert set(_declared_constants().values()) == set(events.ALL_EVENTS)


def test_plan_14_names_present() -> None:
    required = {
        "stage.start",
        "stage.skip",
        "stage.done",
        "stage.fail",
        "blender.spawn",
        "blender.exit",
        "tripo.request",
        "tripo.response",
        "tripo.poll",
        "tripo.credits",
        "verify.route",
        "rfc.result",
        "handoff.sync",
    }
    assert required <= set(events.ALL_EVENTS)
