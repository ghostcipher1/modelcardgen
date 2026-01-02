from datetime import date
from src.modelcardgen.core.models import RiskAssessment
from src.modelcardgen.reports.risk import RiskReportGenerator

risks = [
    RiskAssessment(
        risk_type="Data Bias",
        description="Training data contains historical imbalances with lower accuracy for minority populations. The model was trained on data that underrepresents certain demographic groups.",
        mitigation_strategy="Conduct quarterly fairness audits using stratified test sets. Monitor prediction distributions across demographic groups and document any performance disparities.",
        severity="Medium"
    ),
    RiskAssessment(
        risk_type="Data Drift",
        description="Model was trained on email patterns from 2020-2023. User behavior and spam tactics evolve continuously, and the distribution of incoming email may shift significantly over time.",
        mitigation_strategy="Monitor model performance metrics monthly on recent email samples. If accuracy drops below 94%, trigger immediate retraining. Schedule annual full retraining cycles.",
        severity="Medium"
    ),
    RiskAssessment(
        risk_type="Adversarial Robustness",
        description="Model may be vulnerable to adversarially crafted emails designed to evade detection. Spam operators can adapt to learned patterns and craft emails that bypass the classifier.",
        mitigation_strategy="Implement ensemble methods with multiple models. Maintain a feedback loop for false negatives. Test model against known adversarial techniques quarterly.",
        severity="High"
    ),
    RiskAssessment(
        risk_type="Model Dependency",
        description="Email service depends entirely on this single model for spam filtering. No human review fallback. System failures will directly impact user experience.",
        mitigation_strategy="Implement confidence thresholds for low-confidence predictions. Route uncertain emails to human review. Maintain legacy rule-based filters as backup.",
        severity="High"
    ),
    RiskAssessment(
        risk_type="False Positive Impact",
        description="Misclassifying legitimate emails as spam causes user frustration and potential loss of important messages (invoices, notifications, security alerts).",
        mitigation_strategy="Tune decision threshold to minimize false positives. Implement email recovery features. Monitor false positive rate daily.",
        severity="Medium"
    ),
]

custom_do_not_use = [
    {
        "condition": "Deployed without human review layer",
        "explanation": "All spam filtering decisions should be reviewable by administrators before final blocking."
    },
    {
        "condition": "Used for filtering non-email messages",
        "explanation": "Model was trained exclusively on email text. SMS, chat, and social media may have different characteristics."
    },
    {
        "condition": "Applied to different languages without retraining",
        "explanation": "Model was trained on English-language emails only. Non-English email filtering requires separate validation."
    }
]

generator = RiskReportGenerator()
output_path = generator.generate(
    model_name="Email Spam Classifier",
    model_version="3.2.1",
    risks=risks,
    output_path="EXAMPLE_RISK_REPORT.md",
    report_date=date(2026, 1, 1),
    do_not_use_conditions=custom_do_not_use
)

print(f"Generated: {output_path}")
