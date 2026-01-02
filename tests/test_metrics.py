import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from modelcardgen.core.metrics import MetricsParser, MetricsParsingError
from modelcardgen.core.models import EvaluationMetrics


class TestMetricsParserFromSklearnReport:
    """Tests for parsing scikit-learn classification_report output."""

    def test_valid_sklearn_report(self):
        report = {
            "accuracy": 0.95,
            "weighted avg": {
                "precision": 0.94,
                "recall": 0.93,
                "f1-score": 0.935,
            },
        }
        metrics = MetricsParser.from_sklearn_report(report)
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.94
        assert metrics.recall == 0.93
        assert metrics.f1_score == 0.935

    def test_sklearn_report_missing_accuracy(self):
        report = {
            "weighted avg": {
                "precision": 0.94,
                "recall": 0.93,
                "f1-score": 0.935,
            }
        }
        with pytest.raises(MetricsParsingError, match="Missing required metric"):
            MetricsParser.from_sklearn_report(report)

    def test_sklearn_report_missing_weighted_avg(self):
        report = {"accuracy": 0.95}
        with pytest.raises(MetricsParsingError, match="Missing required metric"):
            MetricsParser.from_sklearn_report(report)

    def test_sklearn_report_missing_metric_in_weighted_avg(self):
        report = {
            "accuracy": 0.95,
            "weighted avg": {
                "precision": 0.94,
            },
        }
        with pytest.raises(MetricsParsingError, match="Missing precision, recall"):
            MetricsParser.from_sklearn_report(report)

    def test_sklearn_report_invalid_type(self):
        with pytest.raises(MetricsParsingError, match="Input must be a dictionary"):
            MetricsParser.from_sklearn_report("not a dict")

    def test_sklearn_report_invalid_metric_value(self):
        report = {
            "accuracy": "not_a_number",
            "weighted avg": {
                "precision": 0.94,
                "recall": 0.93,
                "f1-score": 0.935,
            },
        }
        with pytest.raises(MetricsParsingError, match="Invalid metric value"):
            MetricsParser.from_sklearn_report(report)


class TestMetricsParserFromDict:
    """Tests for parsing metrics from dictionaries."""

    def test_valid_dict_minimal(self):
        metrics_dict = {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1_score": 0.935,
        }
        metrics = MetricsParser.from_dict(metrics_dict)
        assert metrics.accuracy == 0.95
        assert metrics.roc_auc is None
        assert metrics.confusion_matrix is None

    def test_valid_dict_with_optional_fields(self):
        metrics_dict = {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1_score": 0.935,
            "roc_auc": 0.98,
            "confusion_matrix": [[95, 5], [7, 93]],
            "custom_metrics": {"specificity": 0.99},
        }
        metrics = MetricsParser.from_dict(metrics_dict)
        assert metrics.roc_auc == 0.98
        assert metrics.confusion_matrix == [[95, 5], [7, 93]]
        assert metrics.custom_metrics["specificity"] == 0.99

    def test_dict_missing_required_field(self):
        metrics_dict = {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
        }
        with pytest.raises(MetricsParsingError, match="Missing required metrics"):
            MetricsParser.from_dict(metrics_dict)

    def test_dict_invalid_metric_value(self):
        metrics_dict = {
            "accuracy": "invalid",
            "precision": 0.94,
            "recall": 0.93,
            "f1_score": 0.935,
        }
        with pytest.raises(MetricsParsingError, match="Invalid metric value"):
            MetricsParser.from_dict(metrics_dict)

    def test_dict_roc_auc_none(self):
        metrics_dict = {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1_score": 0.935,
            "roc_auc": None,
        }
        metrics = MetricsParser.from_dict(metrics_dict)
        assert metrics.roc_auc is None


class TestMetricsParserFromJsonFile:
    """Tests for parsing metrics from JSON files."""

    def test_valid_json_file(self):
        json_data = {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1_score": 0.935,
        }
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            metrics = MetricsParser.from_json_file(temp_path)
            assert metrics.accuracy == 0.95
        finally:
            Path(temp_path).unlink()

    def test_json_file_not_found(self):
        with pytest.raises(MetricsParsingError, match="File not found"):
            MetricsParser.from_json_file("/nonexistent/path.json")

    def test_invalid_json_file(self):
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with pytest.raises(MetricsParsingError, match="Invalid JSON"):
                MetricsParser.from_json_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestMetricsParserFromConfusionMatrix:
    """Tests for computing metrics from confusion matrices."""

    def test_valid_binary_confusion_matrix_list(self):
        cm = [[95, 5], [7, 93]]
        metrics = MetricsParser.from_confusion_matrix(cm)
        assert isinstance(metrics, EvaluationMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
        assert metrics.confusion_matrix == cm

    def test_valid_binary_confusion_matrix_numpy(self):
        cm = np.array([[95, 5], [7, 93]])
        metrics = MetricsParser.from_confusion_matrix(cm)
        assert metrics.accuracy == pytest.approx(0.94, abs=0.01)

    def test_multiclass_confusion_matrix(self):
        cm = [[90, 5, 5], [3, 92, 5], [2, 6, 92]]
        metrics = MetricsParser.from_confusion_matrix(cm)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1

    def test_non_square_confusion_matrix(self):
        cm = np.array([[95, 5, 3], [7, 93, 5]])
        with pytest.raises(MetricsParsingError, match="must be square"):
            MetricsParser.from_confusion_matrix(cm)

    def test_invalid_confusion_matrix_type(self):
        with pytest.raises(MetricsParsingError, match="Invalid confusion matrix"):
            MetricsParser.from_confusion_matrix("not a matrix")

    def test_confusion_matrix_perfect_classifier(self):
        cm = [[100, 0], [0, 100]]
        metrics = MetricsParser.from_confusion_matrix(cm)
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0

    def test_confusion_matrix_random_classifier(self):
        cm = [[50, 50], [50, 50]]
        metrics = MetricsParser.from_confusion_matrix(cm)
        assert metrics.accuracy == 0.5


class TestEvaluationMetricsValidation:
    """Tests for EvaluationMetrics Pydantic validation."""

    def test_metrics_out_of_range_accuracy(self):
        with pytest.raises(Exception):
            EvaluationMetrics(accuracy=1.5, precision=0.94, recall=0.93, f1_score=0.935)

    def test_metrics_negative_metric(self):
        with pytest.raises(Exception):
            EvaluationMetrics(
                accuracy=0.95, precision=-0.1, recall=0.93, f1_score=0.935
            )

    def test_metrics_valid_bounds(self):
        metrics = EvaluationMetrics(
            accuracy=0.0, precision=1.0, recall=0.5, f1_score=0.75
        )
        assert metrics.accuracy == 0.0
        assert metrics.precision == 1.0
