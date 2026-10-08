"""Run a real, optionally billable policy evaluation and save calculated results.

Usage: python evaluation.py --output evaluation-results.json
This creates a private in-process application session, not a public evaluation API.
"""

import argparse
import json
from pathlib import Path

from backend.core.config import ROOT, Settings
from backend.core.exceptions import AppError
from backend.main import create_app
from backend.services.evaluation_service import EvaluationCase, EvaluationService


def main() -> None:
    """Process the fictional sample and evaluate actual provider answers."""
    parser = argparse.ArgumentParser(description="Run IntelliChat's real-provider evaluation.")
    parser.add_argument("--output", default="evaluation-results.json")
    arguments = parser.parse_args()
    app = create_app(Settings())
    cases = [
        EvaluationCase.model_validate(item)
        for item in json.loads(
            (ROOT / "sample_data/evaluation_dataset.json").read_text(encoding="utf-8")
        )
    ]
    sample = ROOT / "sample_data/company_policy.txt"
    try:
        app.state.document_service.upload("evaluation", [(sample.name, sample.read_bytes())])
        app.state.document_service.process("evaluation")
        result = EvaluationService(app.state.chat_service).evaluate("evaluation", cases)
    except AppError as error:
        parser.exit(1, f"Evaluation stopped: {error.message}\n")
    Path(arguments.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(result['cases'])} measured cases to {arguments.output}")


if __name__ == "__main__":
    main()
