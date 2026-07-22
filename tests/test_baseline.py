"""Phase 0 baseline gate: the fixed eval set reproduces the saved metrics.

BUILD_SPEC §8 Phase 0 gate — "평가셋 재현 가능 + 베이스라인 지표 저장". The
deterministic offline stack (mock provider + ``MockFaithfulness``) must
reproduce ``eval/baseline.json`` exactly, so any drift in routing or grounding
is caught here (and in CI). Regenerate with ``python scripts/snapshot_baseline.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from langconnect_agent.config import Config
from langconnect_agent.evaluation import MockFaithfulness, evaluate

BASELINE_PATH = Path(__file__).resolve().parent.parent / "eval" / "baseline.json"


def _deterministic_report_dict() -> dict:
    report = evaluate(
        config=Config(llm_provider="mock"), faithfulness=MockFaithfulness()
    )
    return report.to_dict()


def test_baseline_file_exists_and_is_well_formed():
    assert BASELINE_PATH.exists(), f"missing Phase 0 baseline: {BASELINE_PATH}"
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert data["faithfulness_metric"] == "MockFaithfulness"
    assert set(data["metrics"]) >= {
        "routing_accuracy",
        "fallback_rate",
        "recovery_rate",
        "per_route_faithfulness",
    }
    assert len(data["cases"]) == 7


def test_current_eval_reproduces_baseline():
    # Phase 0 gate: deterministic stack must match the committed snapshot.
    saved = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert _deterministic_report_dict() == saved
