"""Run a LangSmith experiment over the eval dataset.

    python scripts/run_experiment.py

Registers the dataset if needed, then runs the graph over it with routing /
fallback / faithfulness evaluators. The experiment appears in the LangSmith UI:
Datasets & Experiments -> wayfinder-eval -> Experiments tab. Needs a
LANGSMITH_API_KEY (loaded from .env).
"""

from __future__ import annotations

from langconnect_agent.env import load_env

load_env()

from langconnect_agent.evaluation import (  # noqa: E402
    register_langsmith_dataset,
    run_langsmith_experiment,
)


def main() -> None:
    print(register_langsmith_dataset())
    result = run_langsmith_experiment()
    print(result)


if __name__ == "__main__":
    main()
