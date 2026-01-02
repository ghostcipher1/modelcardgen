import pytest
from modelcardgen.reports.markdown import MarkdownCardGenerator
from modelcardgen.core.models import (
    ModelMetadata,
    DatasetMetadata,
    EvaluationMetrics,
    ModelLimitations,
    UseCaseConstraints,
)


def test_report_generation(tmp_path):
    metadata = ModelMetadata(
        name="TestModel",
        version="1.0.0",
        description="A test classification model.",
        owner="Test Team",
        license="Apache-2.0",
        framework="scikit-learn",
    )

    training_data = DatasetMetadata(
        name="Training Set",
        description="Test training data.",
        size=1000,
    )

    eval_data = DatasetMetadata(
        name="Test Set",
        description="Test evaluation data.",
        size=500,
    )

    metrics = EvaluationMetrics(
        accuracy=0.95,
        precision=0.94,
        recall=0.96,
        f1_score=0.95,
    )

    limitations = ModelLimitations(
        unsuitable_inputs=["None"],
        out_of_scope_uses=["None"],
    )

    constraints = UseCaseConstraints(
        intended_users=["Data Scientists"],
        intended_use_cases=["Classification"],
        prohibited_uses=["None"],
    )

    output_file = tmp_path / "MODEL_CARD.md"
    generator = MarkdownCardGenerator()
    generator.generate(
        metadata=metadata,
        training_data=training_data,
        eval_data=eval_data,
        metrics=metrics,
        limitations=limitations,
        constraints=constraints,
        output_path=str(output_file),
    )

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "# Model Card: TestModel" in content
    assert "95.0%" in content or "0.95" in content
