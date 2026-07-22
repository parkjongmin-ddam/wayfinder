"""Phase 0 baseline snapshot — persist the deterministic eval metrics.

Runs the fixed eval set (BUILD_SPEC §8 Phase 0) through the fully-offline
default stack — mock router + stub sources + ``MockFaithfulness`` — which is
100% reproducible, and writes the metrics to ``eval/baseline.json``. This is the
Phase 0 gate artifact: "평가셋 재현 가능 + 베이스라인 지표 저장".

    python scripts/snapshot_baseline.py           # (re)write eval/baseline.json
    python scripts/snapshot_baseline.py --check    # verify current == saved

Deliberately does NOT read ``.env``: the baseline must be the deterministic mock
stack regardless of a local ``LLM_PROVIDER=ollama`` (or any provider) setting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langconnect_agent.config import Config
from langconnect_agent.evaluation import MockFaithfulness, evaluate

BASELINE_PATH = Path(__file__).resolve().parent.parent / "eval" / "baseline.json"


def deterministic_report():
    """The reproducible baseline: mock provider stack + lexical faithfulness."""
    return evaluate(config=Config(llm_provider="mock"), faithfulness=MockFaithfulness())


def _dump(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    report = deterministic_report()
    data = report.to_dict()

    if "--check" in argv:
        if not BASELINE_PATH.exists():
            print(f"[FAIL] no baseline at {BASELINE_PATH} — run without --check first")
            return 1
        saved = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if saved == data:
            print(f"[OK] current eval matches baseline ({BASELINE_PATH.name})")
            return 0
        print(f"[DRIFT] current eval differs from {BASELINE_PATH.name}")
        return 1

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(_dump(data), encoding="utf-8")
    print(f"[written] {BASELINE_PATH}")
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
