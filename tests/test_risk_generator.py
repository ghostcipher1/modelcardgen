import pytest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from modelcardgen.core.models import RiskAssessment
from modelcardgen.reports.risk import RiskReportGenerator


@pytest.fixture
def sample_risks():
    return [
        RiskAssessment(
            risk_type="Data Bias",
            description="Training data contains historical biases that may affect underrepresented groups.",
            mitigation_strategy="Conduct fairness audits quarterly and monitor prediction distributions.",
            severity="Medium",
        ),
        RiskAssessment(
            risk_type="Data Drift",
            description="Model performance may degrade if customer behavior patterns shift significantly.",
            mitigation_strategy="Monitor model performance monthly with new data and retrain annually.",
            severity="Medium",
        ),
        RiskAssessment(
            risk_type="Security",
            description="Unauthorized access to model predictions could enable gaming or fraud.",
            mitigation_strategy="Implement access controls, audit logging, and rate limiting on model endpoints.",
            severity="High",
        ),
    ]


@pytest.fixture
def high_risk_model_name():
    return "High-Risk Classifier"


@pytest.fixture
def model_version():
    return "1.2.3"


class TestRiskReportGeneratorBasic:
    """Tests for basic risk report generation."""

    def test_generator_initialization(self):
        generator = RiskReportGenerator()
        assert generator.env is not None

    def test_generate_creates_file(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"

            result = generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            assert result.exists()
            assert result == output_path

    def test_generate_file_has_content(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert len(content) > 0


class TestRiskReportGeneratorContent:
    """Tests for generated risk report content."""

    def test_report_contains_model_name(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert high_risk_model_name in content

    def test_report_contains_model_version(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert model_version in content

    def test_report_contains_all_required_sections(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            required_sections = [
                "## Executive Summary",
                "## Risk Classification Framework",
                "## Identified Risks",
                "## Deployment Risk Categories",
                "## Do Not Use If",
                "## Risk Mitigation Requirements",
                "## High-Risk Deployment Scenarios",
                "## Ongoing Risk Management",
                "## Recommendations",
                "## Assessment Metadata",
            ]

            for section in required_sections:
                assert section in content, f"Missing section: {section}"

    def test_report_contains_all_identified_risks(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            for risk in sample_risks:
                assert risk.risk_type in content
                assert risk.description in content
                assert risk.mitigation_strategy in content
                assert risk.severity in content

    def test_report_contains_do_not_use_section(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "Do Not Use If" in content
            assert "The intended use case has changed since evaluation" in content

    def test_report_contains_deployment_risk_categories(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            categories = [
                "### 1. Data Bias and Fairness Risks",
                "### 2. Performance Degradation Risks",
                "### 3. Misuse and Security Risks",
                "### 4. Operational Risks",
            ]

            for category in categories:
                assert category in content

    def test_report_contains_mandatory_mitigations(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            mitigations = [
                "Monitoring and Alerting",
                "Access Control",
                "User Training",
                "Fallback Procedures",
                "Regular Audits",
            ]

            for mitigation in mitigations:
                assert mitigation in content


class TestRiskReportGeneratorRiskClassification:
    """Tests for risk level classification logic."""

    def test_no_risks_classified_as_minimal(self):
        generator = RiskReportGenerator()
        summary = generator._classify_risk_level([])
        assert summary == "minimal documented risk"

    def test_low_severity_only_classified_correctly(self):
        risks = [
            RiskAssessment(
                risk_type="Test",
                description="Test",
                mitigation_strategy="Test",
                severity="Low",
            )
        ]
        generator = RiskReportGenerator()
        summary = generator._classify_risk_level(risks)
        assert summary == "low documented risk"

    def test_medium_severity_classified_correctly(self):
        risks = [
            RiskAssessment(
                risk_type="Test1",
                description="Test",
                mitigation_strategy="Test",
                severity="Low",
            ),
            RiskAssessment(
                risk_type="Test2",
                description="Test",
                mitigation_strategy="Test",
                severity="Medium",
            ),
        ]
        generator = RiskReportGenerator()
        summary = generator._classify_risk_level(risks)
        assert summary == "moderate risk"

    def test_high_severity_classified_as_significant(self):
        risks = [
            RiskAssessment(
                risk_type="Test1",
                description="Test",
                mitigation_strategy="Test",
                severity="Low",
            ),
            RiskAssessment(
                risk_type="Test2",
                description="Test",
                mitigation_strategy="Test",
                severity="Medium",
            ),
            RiskAssessment(
                risk_type="Test3",
                description="Test",
                mitigation_strategy="Test",
                severity="High",
            ),
        ]
        generator = RiskReportGenerator()
        summary = generator._classify_risk_level(risks)
        assert summary == "significant risk"


class TestRiskReportGeneratorMetadata:
    """Tests for assessment metadata and counts."""

    def test_assessment_metadata_includes_risk_counts(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "Total Risks Identified:" in content and "3" in content
            assert "High Severity:" in content and "1" in content
            assert "Medium Severity:" in content and "2" in content
            assert "Low Severity:" in content and "0" in content

    def test_assessment_metadata_includes_date(
        self, sample_risks, high_risk_model_name, model_version
    ):
        report_date = date(2024, 1, 15)
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
                report_date=report_date,
            )

            content = output_path.read_text()
            assert "2024-01-15" in content


class TestRiskReportGeneratorDeterminism:
    """Tests for deterministic output."""

    def test_deterministic_generation(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()
        report_date = date(2024, 1, 15)

        with TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "report1.md"
            path2 = Path(tmpdir) / "report2.md"

            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=path1,
                report_date=report_date,
            )

            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=path2,
                report_date=report_date,
            )

            assert path1.read_text() == path2.read_text()


class TestRiskReportGeneratorEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_risks_list(self, high_risk_model_name, model_version):
        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=[],
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "minimal documented risk" in content
            assert "Total Risks Identified:" in content and "0" in content

    def test_report_with_custom_do_not_use_conditions(
        self, sample_risks, high_risk_model_name, model_version
    ):
        custom_conditions = [
            {
                "condition": "Used in safety-critical systems",
                "explanation": "Model has not been validated for life-critical applications.",
            },
            {
                "condition": "Deployed without human oversight",
                "explanation": "All predictions must be reviewed by qualified domain experts.",
            },
        ]

        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
                do_not_use_conditions=custom_conditions,
            )

            content = output_path.read_text()
            for condition in custom_conditions:
                assert condition["condition"] in content
                assert condition["explanation"] in content

    def test_report_with_all_high_severity_risks(
        self, high_risk_model_name, model_version
    ):
        high_severity_risks = [
            RiskAssessment(
                risk_type="Critical Risk 1",
                description="This poses a critical threat.",
                mitigation_strategy="Immediate action required.",
                severity="High",
            ),
            RiskAssessment(
                risk_type="Critical Risk 2",
                description="This also poses a critical threat.",
                mitigation_strategy="Immediate action required.",
                severity="High",
            ),
        ]

        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=high_severity_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "significant risk" in content
            assert "High Severity:" in content and "2" in content
            assert "Medium Severity:" in content and "0" in content

    def test_report_with_special_characters(self, model_version):
        special_name = 'Model with "quotes" & <special> chars'
        risks = [
            RiskAssessment(
                risk_type="Risk with <tags>",
                description="Description with & ampersands",
                mitigation_strategy='Mitigation with "quotes"',
                severity="Low",
            )
        ]

        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=special_name,
                model_version=model_version,
                risks=risks,
                output_path=output_path,
            )

            assert output_path.exists()
            content = output_path.read_text()
            assert len(content) > 0

    def test_default_report_date_is_today(
        self, sample_risks, high_risk_model_name, model_version
    ):
        generator = RiskReportGenerator()
        today = date.today()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=sample_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert str(today) in content


class TestRiskReportGeneratorIntegration:
    """Integration tests with multiple risk types."""

    def test_bias_risks_categorized_correctly(
        self, high_risk_model_name, model_version
    ):
        bias_risks = [
            RiskAssessment(
                risk_type="Data Bias",
                description="Training data contains demographic imbalances.",
                mitigation_strategy="Audit fairness metrics quarterly.",
                severity="Medium",
            ),
            RiskAssessment(
                risk_type="Language Bias",
                description="Model trained primarily on English text.",
                mitigation_strategy="Test on non-English samples.",
                severity="Low",
            ),
        ]

        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=bias_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "### 1. Data Bias and Fairness Risks" in content
            assert "Data Bias" in content
            assert "Language Bias" in content

    def test_degradation_risks_categorized_correctly(
        self, high_risk_model_name, model_version
    ):
        degradation_risks = [
            RiskAssessment(
                risk_type="Data Drift",
                description="User behavior may shift over time.",
                mitigation_strategy="Monitor monthly and retrain annually.",
                severity="Medium",
            ),
            RiskAssessment(
                risk_type="Concept Drift",
                description="Market conditions may change model relevance.",
                mitigation_strategy="Quarterly reviews of model relevance.",
                severity="Medium",
            ),
        ]

        generator = RiskReportGenerator()

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "RISK_REPORT.md"
            generator.generate(
                model_name=high_risk_model_name,
                model_version=model_version,
                risks=degradation_risks,
                output_path=output_path,
            )

            content = output_path.read_text()
            assert "### 2. Performance Degradation Risks" in content
            assert "Data Drift" in content
