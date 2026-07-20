"""Phase 4 evaluation runner — offline differentiator metrics on mock data.

    python scripts/run_eval.py

Prints routing accuracy, fallback firing/recovery rate, and per-route
faithfulness over the fixed eval set (BUILD_SPEC §4 / Phase 4 gate). Also shows
whether the eval set would register as a LangSmith dataset (needs a key).
"""

from __future__ import annotations

from langconnect_agent.env import load_env

load_env()  # pull .env (keys, LangSmith config) into the environment

from langconnect_agent.evaluation import (  # noqa: E402
    evaluate,
    register_langsmith_dataset,
)


def main() -> None:
    report = evaluate()
    print(report.summary())
    print()
    print("per-case:")
    for r in report.results:
        flag = "OK " if r.routed_correctly else "MISS"
        fb = f"->web" if r.fell_back else "    "
        print(
            f"  [{flag}] {r.actual_route:9s} {fb} "
            f"faith={r.faithfulness:.3f}  {r.case.query!r}"
        )
    print()
    print(register_langsmith_dataset())


if __name__ == "__main__":
    main()
