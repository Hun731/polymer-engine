"""The monitor must survive an open-ended campaign (`duration_hours: null`).

It did not. `float(cfg["campaign"]["duration_hours"])` raised at startup against the
open-ended config, so the monitor died immediately while the driver ran on -- leaving a
live campaign unobserved and, from the outside, looking as though the campaign itself
had failed. Two further sites then formatted `remaining_hours` as a number.

Open-ended is the project default now, so these are the cases that matter.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _monitor():
    spec = importlib.util.spec_from_file_location(
        "monitor_campaign",
        Path(__file__).resolve().parents[2] / "scripts" / "monitor_campaign.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def campaign(tmp_path: Path) -> Path:
    import time

    (tmp_path / "campaign_state.json").write_text(json.dumps({
        "started_at": "2026-09-01T00:00:00+00:00",
        "started_epoch": time.time() - 3600.0,   # one hour in
        "results": {}, "queue": [], "failed": [], "completed": [],
        "attempts": {}, "decisions": [], "iteration": 0, "stopped_reason": None,
    }))
    return tmp_path


def _cfg(duration):
    """The real shipped config with only the duration changed.

    Built from the file rather than hand-written, so the test cannot pass against a
    config shape the campaign no longer uses.
    """
    import yaml

    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "configs" / "campaign_100ns.yaml").read_text())
    cfg["campaign"]["duration_hours"] = duration
    return cfg


def test_an_open_ended_campaign_has_no_remaining_time(campaign: Path) -> None:
    status = _monitor().build_status(campaign, _cfg(None), None)
    assert status["budget_hours"] is None
    # None, not zero: having no deadline is not the same as having no time left.
    assert status["remaining_hours"] is None
    assert status["open_ended"] is True


def test_a_time_boxed_campaign_still_reports_a_budget(campaign: Path) -> None:
    status = _monitor().build_status(campaign, _cfg(96.0), 96.0)
    assert status["budget_hours"] == 96.0
    assert status["remaining_hours"] is not None
    assert status["open_ended"] is False


def test_the_markdown_renders_without_a_deadline(campaign: Path) -> None:
    module = _monitor()
    rendered = module.render_markdown(module.build_status(campaign, _cfg(None), None))
    assert "open-ended" in rendered
    assert "no deadline" in rendered


def test_the_markdown_still_renders_a_deadline_when_there_is_one(campaign: Path) -> None:
    module = _monitor()
    rendered = module.render_markdown(module.build_status(campaign, _cfg(96.0), 96.0))
    assert "96" in rendered
    assert "open-ended" not in rendered


@pytest.mark.parametrize("duration", [None, 96.0])
def test_startup_parses_either_kind_of_budget(duration) -> None:
    """The exact expression that killed the monitor at line 280."""
    raw = _cfg(duration)["campaign"].get("duration_hours")
    budget = None if raw is None else float(raw)
    assert budget == (None if duration is None else 96.0)
