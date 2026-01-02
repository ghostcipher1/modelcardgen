from datetime import date
from src.modelcardgen.core.models import (
    DatasetMetadata,
    EvaluationMetrics,
    ModelLimitations,
    ModelMetadata,
    RiskAssessment,
    UseCaseConstraints,
)
from src.modelcardgen.reports.markdown import MarkdownCardGenerator

metadata = ModelMetadata(
    name="Email Spam Classifier",
    version="2.1.0",
    description="A machine learning model that classifies emails as spam or legitimate based on textual features and header information.",
    owner="Machine Learning Team",
    license="Apache-2.0",
    framework="scikit-learn"
)

training_data = DatasetMetadata(
    name="Enron Email Corpus",
    description="A large collection of real email messages with manually labeled spam and legitimate categories.",
    size=755000,
    features=["subject_line", "body_text", "sender_domain", "header_features"],
    target="spam_label"
)

eval_data = DatasetMetadata(
    name="Recent Email Dataset",
    description="A holdout test set from more recent emails to evaluate temporal robustness.",
    size=50000,
    features=["subject_line", "body_text", "sender_domain", "header_features"],
    target="spam_label"
)

metrics = EvaluationMetrics(
    accuracy=0.963,
    precision=0.951,
    recall=0.945,
    f1_score=0.948,
    roc_auc=0.985,
    confusion_matrix=[[47500, 1500], [2500, 500], [1000, 45500]],
    custom_metrics={"specificity": 0.969, "false_negative_rate": 0.055}
)

limitations = ModelLimitations(
    unsuitable_inputs=[
        "Non-English emails",
        "Encrypted or binary content",
        "Severely truncated messages"
    ],
    environmental_constraints="Requires Python 3.10+. Typical inference time: <50ms per email.",
    out_of_scope_uses=[
        "Real-time filtering without human review",
        "Automated account suspension"
    ]
)

constraints = UseCaseConstraints(
    intended_users=["Email administrators", "IT security teams", "Email service providers"],
    intended_use_cases=[
        "Spam detection for user inboxes",
        "Training data for downstream models",
        "Email security analytics"
    ],
    prohibited_uses=[
        "Blocking legitimate user email",
        "Discriminatory filtering based on sender identity"
    ]
)

risks = [
    RiskAssessment(
        risk_type="Domain-Specific Performance",
        description="Model was trained on Enron corpus (2000s) and may not capture modern spam patterns.",
        mitigation_strategy="Retrain quarterly with recent email data. Monitor performance metrics continuously.",
        severity="Medium"
    ),
    RiskAssessment(
        risk_type="Language Bias",
        description="Training data contains primarily English emails. Non-English emails may have degraded performance.",
        mitigation_strategy="Collect and evaluate performance on non-English email samples.",
        severity="Low"
    )
]

generator = MarkdownCardGenerator()
output_path = generator.generate(
    metadata=metadata,
    training_data=training_data,
    eval_data=eval_data,
    metrics=metrics,
    limitations=limitations,
    constraints=constraints,
    risks=risks,
    output_path="EXAMPLE_MODEL_CARD.md"
)

print(f"Generated: {output_path}")
