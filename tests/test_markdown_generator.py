import pytest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from modelcardgen.core.models import (
    DatasetMetadata,
    EvaluationMetrics,
    ModelCard,
    ModelLimitations,
    ModelMetadata,
    RiskAssessment,
    UseCaseConstraints,
)
from modelcardgen.reports.markdown import MarkdownCardGenerator


@pytest.fixture
def sample_metadata():
    return ModelMetadata(
        name="Customer Churn Classifier",
        version="1.0.0",
        description="A binary classification model that predicts customer churn likelihood based on account features.",
        owner="Data Science Team",
        license="MIT",
        release_date=date(2024, 1, 15),
        framework="scikit-learn",
    )


@pytest.fixture
def sample_training_data():
    return DatasetMetadata(
        name="Customer Dataset v2",
        description="Historical customer data with churn labels collected over 12 months.",
        source_url=None,
        size=50000,
        features=[
            "account_age",
            "monthly_charges",
            "total_charges",
            "customer_service_calls",
        ],
        target="churn",
    )


@pytest.fixture
def sample_eval_data():
    return DatasetMetadata(
        name="Customer Test Set",
        description="Holdout test set from the same time period as training data.",
        source_url=None,
        size=10000,
        features=[
            "account_age",
            "monthly_charges",
            "total_charges",
            "customer_service_calls",
        ],
        target="churn",
    )


@pytest.fixture
def sample_metrics():
    return EvaluationMetrics(
        accuracy=0.87,
        precision=0.85,
        recall=0.82,
        f1_score=0.835,
        roc_auc=0.92,
        confusion_matrix=[[8400, 600], [1200, 8800]],
        custom_metrics={"specificity": 0.93, "false_positive_rate": 0.07},
    )


@pytest.fixture
def sample_limitations():
    return ModelLimitations(
        unsuitable_inputs=[
            "Data from new customers (account age < 1 month)",
            "Account data with missing values",
            "Non-standard account types (e.g., enterprise contracts)",
        ],
        environmental_constraints="Requires Python 3.10+ and scikit-learn >= 1.0.0",
        out_of_scope_uses=[
            "Predicting customer lifetime value",
            "Identifying fraud cases",
            "Billing determination",
        ],
    )


@pytest.fixture
def sample_constraints():
    return UseCaseConstraints(
        intended_users=[
            "Customer retention team",
            "Business analysts",
            "Account managers",
        ],
        intended_use_cases=[
            "Identifying at-risk customers for proactive retention efforts",
            "Resource allocation for customer success interventions",
            "Business intelligence and trend analysis",
        ],
        prohibited_uses=[
            "Automated termination of customer accounts without human review",
            "Discrimination based on protected characteristics",
            "Sharing predictions with third parties without consent",
        ],
    )


@pytest.fixture
def sample_risks():
    return [
        RiskAssessment(
            risk_type="Class Imbalance",
            description="The dataset contains a 45% positive class ratio, which may affect minority class predictions.",
            mitigation_strategy="Used weighted loss functions during training to account for class imbalance.",
            severity="Low",
        ),
        RiskAssessment(
            risk_type="Data Drift",
            description="Model performance may degrade if customer behavior patterns shift over time.",
            mitigation_strategy="Monitor model performance monthly and retrain annually with new data.",
            severity="Medium",
        ),
    ]


@pytest.fixture
def sample_model_card(
    sample_metadata,
    sample_training_data,
    sample_eval_data,
    sample_metrics,
    sample_limitations,
    sample_constraints,
    sample_risks,
):
    return ModelCard(
        metadata=sample_metadata,
        training_data=sample_training_data,
        eval_data=sample_eval_data,
        metrics=sample_metrics,
        risks=sample_risks,
        limitations=sample_limitations,
        constraints=sample_constraints,
    )


class TestMarkdownCardGeneratorBasic:
    """Tests for basic model card generation."""

    def test_generator_initialization(self):
        generator = MarkdownCardGenerator()
        assert generator.env is not None

    def test_generate_creates_file(
        self,
        sample_metadata,
        sample_training_data,
        sample_eval_data,
        sample_metrics,
        sample_limitations,
        sample_constraints,
    ):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"

            result = generator.generate(
                metadata=sample_metadata,
                training_data=sample_training_data,
                eval_data=sample_eval_data,
                metrics=sample_metrics,
                limitations=sample_limitations,
                constraints=sample_constraints,
                output_path=output_path,
            )

            assert result.exists()
            assert result == output_path

    def test_generate_from_model_card_creates_file(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"

            result = generator.generate_from_model_card(
                sample_model_card, output_path=output_path
            )

            assert result.exists()
            assert result == output_path


class TestMarkdownCardGeneratorContent:
    """Tests for generated model card content."""

    def test_model_card_contains_model_name(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            assert "Customer Churn Classifier" in content
            assert sample_model_card.metadata.name in content

    def test_model_card_contains_all_required_sections(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            required_sections = [
                "## Model Overview",
                "## Intended Use",
                "## Dataset Summary",
                "## Performance Summary",
                "## Known Limitations",
                "## Ethical Considerations and Risks",
                "## Out-of-Scope Uses",
            ]

            for section in required_sections:
                assert section in content, f"Missing section: {section}"

    def test_model_card_contains_metadata(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            assert sample_model_card.metadata.version in content
            assert sample_model_card.metadata.owner in content
            assert sample_model_card.metadata.license in content
            assert sample_model_card.metadata.framework in content

    def test_model_card_contains_metrics(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            assert "Accuracy" in content
            assert "Precision" in content
            assert "Recall" in content
            assert "F1-Score" in content
            assert "0.870" in content  # accuracy formatted

    def test_model_card_contains_intended_use_cases(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            for use_case in sample_model_card.constraints.intended_use_cases:
                assert use_case in content

    def test_model_card_contains_limitations(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            for limitation in sample_model_card.limitations.unsuitable_inputs:
                assert limitation in content

    def test_model_card_contains_risks(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            for risk in sample_model_card.risks:
                assert risk.risk_type in content
                assert risk.description in content
                assert risk.mitigation_strategy in content

    def test_model_card_contains_prohibited_uses(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            for prohibited_use in sample_model_card.constraints.prohibited_uses:
                assert prohibited_use in content


class TestMarkdownCardGeneratorDeterminism:
    """Tests for deterministic output (same input → same output)."""

    def test_deterministic_generation(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "card1.md"
            path2 = Path(tmpdir) / "card2.md"

            generator.generate_from_model_card(sample_model_card, path1)
            generator.generate_from_model_card(sample_model_card, path2)

            content1 = path1.read_text()
            content2 = path2.read_text()

            assert content1 == content2

    def test_deterministic_with_optional_fields(self):
        """Test determinism when optional fields are present or absent."""
        metadata = ModelMetadata(
            name="Test Model",
            version="1.0.0",
            description="Test model",
            owner="Test Owner",
            license="MIT",
            framework="sklearn",
        )

        training_data = DatasetMetadata(
            name="Training Data", description="Training data", target="target"
        )

        eval_data = DatasetMetadata(
            name="Eval Data", description="Eval data", target="target"
        )

        metrics = EvaluationMetrics(
            accuracy=0.95, precision=0.94, recall=0.93, f1_score=0.935
        )

        limitations = ModelLimitations(
            unsuitable_inputs=["bad data"], out_of_scope_uses=["bad use"]
        )

        constraints = UseCaseConstraints(
            intended_users=["user1"],
            intended_use_cases=["use1"],
            prohibited_uses=["prohibited1"],
        )

        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "card1.md"
            path2 = Path(tmpdir) / "card2.md"

            generator.generate(
                metadata=metadata,
                training_data=training_data,
                eval_data=eval_data,
                metrics=metrics,
                limitations=limitations,
                constraints=constraints,
                risks=[],
                output_path=path1,
            )

            generator.generate(
                metadata=metadata,
                training_data=training_data,
                eval_data=eval_data,
                metrics=metrics,
                limitations=limitations,
                constraints=constraints,
                risks=[],
                output_path=path2,
            )

            assert path1.read_text() == path2.read_text()


class TestMarkdownCardGeneratorOptionalFields:
    """Tests for handling optional fields."""

    def test_optional_roc_auc_included_when_present(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            assert "ROC AUC" in content
            assert "0.920" in content

    def test_optional_roc_auc_excluded_when_absent(
        self,
        sample_metadata,
        sample_training_data,
        sample_eval_data,
        sample_limitations,
        sample_constraints,
    ):
        metrics_without_roc = EvaluationMetrics(
            accuracy=0.87, precision=0.85, recall=0.82, f1_score=0.835
        )

        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate(
                metadata=sample_metadata,
                training_data=sample_training_data,
                eval_data=sample_eval_data,
                metrics=metrics_without_roc,
                limitations=sample_limitations,
                constraints=sample_constraints,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "ROC AUC" not in content or "None" in content

    def test_confusion_matrix_included_when_present(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            assert "Confusion Matrix" in content

    def test_environmental_constraints_included_when_present(self, sample_model_card):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate_from_model_card(sample_model_card, output_path)

            content = output_path.read_text()
            assert sample_model_card.limitations.environmental_constraints in content


class TestMarkdownCardGeneratorEdgeCases:
    """Tests for edge cases and special formatting."""

    def test_empty_risks_list(
        self,
        sample_metadata,
        sample_training_data,
        sample_eval_data,
        sample_metrics,
        sample_limitations,
        sample_constraints,
    ):
        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate(
                metadata=sample_metadata,
                training_data=sample_training_data,
                eval_data=sample_eval_data,
                metrics=sample_metrics,
                limitations=sample_limitations,
                constraints=sample_constraints,
                risks=[],
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "Ethical Considerations and Risks" in content

    def test_special_characters_in_description(self):
        metadata = ModelMetadata(
            name='Model with "special" & chars',
            version="1.0.0",
            description="Model for <testing> special chars & symbols",
            owner="Test Owner",
            license="MIT",
            framework="sklearn",
        )

        training_data = DatasetMetadata(
            name="Data", description="Data with special chars", target="target"
        )

        eval_data = DatasetMetadata(name="Data", description="Data", target="target")

        metrics = EvaluationMetrics(
            accuracy=0.95, precision=0.94, recall=0.93, f1_score=0.935
        )

        limitations = ModelLimitations(
            unsuitable_inputs=["data"], out_of_scope_uses=["use"]
        )

        constraints = UseCaseConstraints(
            intended_users=["user"],
            intended_use_cases=["use"],
            prohibited_uses=["prohibited"],
        )

        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate(
                metadata=metadata,
                training_data=training_data,
                eval_data=eval_data,
                metrics=metrics,
                limitations=limitations,
                constraints=constraints,
                output_path=output_path,
            )

            assert output_path.exists()
            content = output_path.read_text()
            assert len(content) > 0

    def test_very_high_accuracy_formatting(self):
        metadata = ModelMetadata(
            name="Perfect Model",
            version="1.0.0",
            description="A model with perfect accuracy",
            owner="Test",
            license="MIT",
            framework="sklearn",
        )

        training_data = DatasetMetadata(
            name="Data", description="Data", target="target"
        )
        eval_data = DatasetMetadata(name="Data", description="Data", target="target")

        metrics = EvaluationMetrics(
            accuracy=1.0, precision=1.0, recall=1.0, f1_score=1.0
        )

        limitations = ModelLimitations(
            unsuitable_inputs=["data"], out_of_scope_uses=["use"]
        )

        constraints = UseCaseConstraints(
            intended_users=["user"],
            intended_use_cases=["use"],
            prohibited_uses=["prohibited"],
        )

        generator = MarkdownCardGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "MODEL_CARD.md"
            generator.generate(
                metadata=metadata,
                training_data=training_data,
                eval_data=eval_data,
                metrics=metrics,
                limitations=limitations,
                constraints=constraints,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "100.0%" in content
