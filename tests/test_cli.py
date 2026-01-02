import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from click.testing import CliRunner

from modelcardgen.cli.main import main


@pytest.fixture
def sample_metrics_data():
    return {
        "accuracy": 0.92,
        "precision": 0.90,
        "recall": 0.94,
        "f1_score": 0.92,
        "model_name": "Test Classifier",
        "model_version": "1.0.0",
        "model_description": "A test classification model.",
        "model_owner": "Test Team",
        "model_license": "MIT",
        "model_framework": "scikit-learn",
        "training_data_name": "Training Set",
        "training_data_description": "Test training data.",
        "training_data_size": 5000,
        "eval_data_name": "Test Set",
        "eval_data_description": "Test evaluation data.",
        "eval_data_size": 1000,
        "intended_use_cases": ["Classification"],
        "intended_users": ["Data Scientists"],
        "unsuitable_inputs": ["Null values"],
        "out_of_scope_uses": ["Real-time classification"],
        "prohibited_uses": ["Discriminatory use"],
        "risks": [
            {
                "risk_type": "Data Bias",
                "description": "Training data may contain bias.",
                "mitigation_strategy": "Monitor fairness metrics.",
                "severity": "Medium",
            }
        ],
    }


@pytest.fixture
def runner():
    return CliRunner()


class TestCliGenerate:
    """Tests for the generate command."""

    def test_generate_with_valid_metrics(self, runner, sample_metrics_data):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            output_dir = Path(tmpdir) / "output"

            with open(metrics_file, "w") as f:
                json.dump(sample_metrics_data, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--metrics",
                    str(metrics_file),
                    "--output-dir",
                    str(output_dir),
                ],
            )

            assert result.exit_code == 0
            assert "Successfully generated" in result.output
            assert (output_dir / "MODEL_CARD.md").exists()
            assert (output_dir / "RISK_REPORT.md").exists()

    def test_generate_creates_output_dir(self, runner, sample_metrics_data):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            output_dir = Path(tmpdir) / "nonexistent" / "nested" / "dir"

            with open(metrics_file, "w") as f:
                json.dump(sample_metrics_data, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--metrics",
                    str(metrics_file),
                    "--output-dir",
                    str(output_dir),
                ],
            )

            assert result.exit_code == 0
            assert output_dir.exists()

    def test_generate_without_metrics_fails(self, runner):
        result = runner.invoke(main, ["generate", "--output-dir", "."])
        assert result.exit_code != 0
        assert "Missing option '--metrics'" in result.output

    def test_generate_with_missing_metrics_file(self, runner):
        result = runner.invoke(
            main,
            ["generate", "--metrics", "/nonexistent/path.json"],
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_generate_with_invalid_json(self, runner):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "bad.json"
            metrics_file.write_text("{ invalid json }")

            result = runner.invoke(
                main,
                ["generate", "--metrics", str(metrics_file)],
            )

            assert result.exit_code != 0

    def test_generate_with_missing_required_metrics(self, runner):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            incomplete_data = {"accuracy": 0.9}

            with open(metrics_file, "w") as f:
                json.dump(incomplete_data, f)

            result = runner.invoke(
                main,
                ["generate", "--metrics", str(metrics_file)],
            )

            assert result.exit_code == 2

    def test_generate_with_confusion_matrix(self, runner, sample_metrics_data):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            cm_file = Path(tmpdir) / "cm.json"
            output_dir = Path(tmpdir) / "output"

            with open(metrics_file, "w") as f:
                json.dump(sample_metrics_data, f)

            with open(cm_file, "w") as f:
                json.dump(
                    {"confusion_matrix": [[400, 60], [40, 500]]},
                    f,
                )

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--metrics",
                    str(metrics_file),
                    "--confusion-matrix",
                    str(cm_file),
                    "--output-dir",
                    str(output_dir),
                ],
            )

            assert result.exit_code == 0
            assert (output_dir / "MODEL_CARD.md").exists()

    def test_generate_with_format_flag(self, runner, sample_metrics_data):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            output_dir = Path(tmpdir) / "output"

            with open(metrics_file, "w") as f:
                json.dump(sample_metrics_data, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--metrics",
                    str(metrics_file),
                    "--output-dir",
                    str(output_dir),
                    "--format",
                    "md",
                ],
            )

            assert result.exit_code == 0

    def test_generate_output_contains_interpretations(
        self, runner, sample_metrics_data
    ):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            output_dir = Path(tmpdir) / "output"

            with open(metrics_file, "w") as f:
                json.dump(sample_metrics_data, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--metrics",
                    str(metrics_file),
                    "--output-dir",
                    str(output_dir),
                ],
            )

            assert result.exit_code == 0

            model_card = (output_dir / "MODEL_CARD.md").read_text()
            assert "Metric Interpretation" in model_card


class TestCliValidate:
    """Tests for the validate command."""

    def test_validate_with_good_metrics(self, runner, sample_metrics_data):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"

            with open(metrics_file, "w") as f:
                json.dump(sample_metrics_data, f)

            result = runner.invoke(
                main,
                ["validate", "--metrics", str(metrics_file)],
            )

            assert result.exit_code == 0
            assert "Validation passed" in result.output

    def test_validate_with_missing_metrics_file(self, runner):
        result = runner.invoke(
            main,
            ["validate", "--metrics", "/nonexistent/path.json"],
        )
        assert result.exit_code != 0

    def test_validate_without_metrics_fails(self, runner):
        result = runner.invoke(main, ["validate"])
        assert result.exit_code != 0
        assert "Missing option '--metrics'" in result.output

    def test_validate_with_invalid_json(self, runner):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "bad.json"
            metrics_file.write_text("{ invalid json }")

            result = runner.invoke(
                main,
                ["validate", "--metrics", str(metrics_file)],
            )

            assert result.exit_code != 0

    def test_validate_with_missing_required_metrics(self, runner):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            incomplete_data = {"accuracy": 0.9}

            with open(metrics_file, "w") as f:
                json.dump(incomplete_data, f)

            result = runner.invoke(
                main,
                ["validate", "--metrics", str(metrics_file)],
            )

            assert result.exit_code == 1

    def test_validate_with_confusion_matrix(self, runner, sample_metrics_data):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            cm_file = Path(tmpdir) / "cm.json"

            with open(metrics_file, "w") as f:
                json.dump(sample_metrics_data, f)

            with open(cm_file, "w") as f:
                json.dump(
                    {"confusion_matrix": [[400, 60], [40, 500]]},
                    f,
                )

            result = runner.invoke(
                main,
                [
                    "validate",
                    "--metrics",
                    str(metrics_file),
                    "--confusion-matrix",
                    str(cm_file),
                ],
            )

            assert result.exit_code == 0

    def test_validate_strict_mode_with_good_metrics(self, runner, sample_metrics_data):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"

            with open(metrics_file, "w") as f:
                json.dump(sample_metrics_data, f)

            result = runner.invoke(
                main,
                ["validate", "--metrics", str(metrics_file), "--strict"],
            )

            assert result.exit_code == 0

    def test_validate_strict_mode_with_low_metrics(self, runner):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            poor_metrics = {
                "accuracy": 0.55,
                "precision": 0.50,
                "recall": 0.48,
                "f1_score": 0.49,
            }

            with open(metrics_file, "w") as f:
                json.dump(poor_metrics, f)

            result = runner.invoke(
                main,
                ["validate", "--metrics", str(metrics_file), "--strict"],
            )

            assert result.exit_code == 2
            assert "Validation failed" in result.output

    def test_validate_reports_metric_issues(self, runner):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            metrics_with_issues = {
                "accuracy": 0.50,
                "precision": 0.40,
                "recall": 0.30,
                "f1_score": 0.35,
            }

            with open(metrics_file, "w") as f:
                json.dump(metrics_with_issues, f)

            result = runner.invoke(
                main,
                ["validate", "--metrics", str(metrics_file)],
            )

            assert result.exit_code == 2
            assert "Metric Interpretation Analysis" in result.output


class TestCliHelp:
    """Tests for help text and command documentation."""

    def test_main_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "CLEAR" in result.output
        assert (
            "Model Cards" in result.output
            or "model documentation" in result.output.lower()
        )

    def test_generate_help(self, runner):
        result = runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--metrics" in result.output
        assert "--output-dir" in result.output
        assert "--format" in result.output
        assert "--strict" in result.output

    def test_validate_help(self, runner):
        result = runner.invoke(main, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--metrics" in result.output
        assert "--strict" in result.output


class TestCliExitCodes:
    """Tests for proper exit codes for CI usage."""

    def test_generate_success_exit_code(self, runner, sample_metrics_data):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            output_dir = Path(tmpdir) / "output"

            with open(metrics_file, "w") as f:
                json.dump(sample_metrics_data, f)

            result = runner.invoke(
                main,
                [
                    "generate",
                    "--metrics",
                    str(metrics_file),
                    "--output-dir",
                    str(output_dir),
                ],
            )

            assert result.exit_code == 0

    def test_validate_success_exit_code(self, runner, sample_metrics_data):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"

            with open(metrics_file, "w") as f:
                json.dump(sample_metrics_data, f)

            result = runner.invoke(
                main,
                ["validate", "--metrics", str(metrics_file)],
            )

            assert result.exit_code == 0

    def test_validation_failure_exit_code(self, runner):
        with TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "metrics.json"
            invalid_data = {"accuracy": 0.5, "precision": 0.4}

            with open(metrics_file, "w") as f:
                json.dump(invalid_data, f)

            result = runner.invoke(
                main,
                ["generate", "--metrics", str(metrics_file)],
            )

            assert result.exit_code == 2

    def test_file_error_exit_code(self, runner):
        result = runner.invoke(
            main,
            ["generate", "--metrics", "/nonexistent/metrics.json"],
        )

        assert result.exit_code == 2
