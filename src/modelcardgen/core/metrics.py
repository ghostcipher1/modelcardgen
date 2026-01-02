import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import ValidationError

from modelcardgen.core.models import EvaluationMetrics

__all__ = [
    "MetricsParsingError",
    "MetricsParser",
]


class MetricsParsingError(Exception):
    """Raised when metrics cannot be parsed or validated."""

    pass


class MetricsParser:
    """
    Parses and normalizes metrics from various sources into EvaluationMetrics.

    Supports:
    - Scikit-learn classification_report output
    - Custom metrics from JSON files or dicts
    - Confusion matrices as numpy arrays or nested lists
    
    **API Stability**: Stable. Public API for metrics parsing and validation.
    All public methods (from_sklearn_report, from_json_file, from_dict, from_confusion_matrix)
    are guaranteed to remain compatible within v0.x versions.
    """

    @staticmethod
    def from_sklearn_report(report_dict: Dict[str, Any]) -> EvaluationMetrics:
        """
        Convert scikit-learn classification_report output to EvaluationMetrics.

        Args:
            report_dict: Dictionary returned by sklearn.metrics.classification_report(output_dict=True)

        Returns:
            EvaluationMetrics instance

        Raises:
            MetricsParsingError: If required metrics are missing or invalid
        """
        if not isinstance(report_dict, dict):
            raise MetricsParsingError("Input must be a dictionary.")

        try:
            accuracy = report_dict.get("accuracy")
            if accuracy is None:
                raise KeyError("accuracy")

            weighted_metrics = report_dict.get("weighted avg", {})
            if not weighted_metrics:
                raise KeyError("weighted avg")

            precision = weighted_metrics.get("precision")
            recall = weighted_metrics.get("recall")
            f1_score = weighted_metrics.get("f1-score")

            if any(v is None for v in [precision, recall, f1_score]):
                raise KeyError("Missing precision, recall, or f1-score in weighted avg")

            return EvaluationMetrics(
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1_score),
                roc_auc=None,
                confusion_matrix=None,
                custom_metrics={},
            )
        except KeyError as e:
            raise MetricsParsingError(
                f"Missing required metric: {e}. "
                f"Expected sklearn classification_report with 'accuracy' and 'weighted avg' keys."
            )
        except (ValueError, TypeError) as e:
            raise MetricsParsingError(f"Invalid metric value type: {e}")

    @staticmethod
    def from_json_file(file_path: Union[str, Path]) -> EvaluationMetrics:
        """
        Load metrics from a JSON file and normalize to EvaluationMetrics.

        JSON structure should map to EvaluationMetrics fields:
        {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "f1_score": 0.935,
            "roc_auc": 0.98,
            "confusion_matrix": [[100, 5], [3, 200]],
            "custom_metrics": {"specificity": 0.99}
        }

        Args:
            file_path: Path to JSON file

        Returns:
            EvaluationMetrics instance

        Raises:
            MetricsParsingError: If file cannot be read or JSON is invalid
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise MetricsParsingError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise MetricsParsingError(f"Invalid JSON in {file_path}: {e}")

        return MetricsParser.from_dict(data)

    @staticmethod
    def from_dict(metrics_dict: Dict[str, Any]) -> EvaluationMetrics:
        """
        Create EvaluationMetrics from a dictionary with flexible field mapping.

        Accepts partial data; required fields are accuracy, precision, recall, f1_score.
        Optional fields: roc_auc, confusion_matrix, custom_metrics.

        Args:
            metrics_dict: Dictionary with metric values

        Returns:
            EvaluationMetrics instance

        Raises:
            MetricsParsingError: If required metrics are missing or invalid
        """
        required = {"accuracy", "precision", "recall", "f1_score"}
        provided = set(metrics_dict.keys())

        missing = required - provided
        if missing:
            raise MetricsParsingError(
                f"Missing required metrics: {', '.join(missing)}. "
                f"Required: {', '.join(required)}"
            )

        try:
            return EvaluationMetrics(
                accuracy=float(metrics_dict["accuracy"]),
                precision=float(metrics_dict["precision"]),
                recall=float(metrics_dict["recall"]),
                f1_score=float(metrics_dict["f1_score"]),
                roc_auc=(
                    float(metrics_dict["roc_auc"])
                    if metrics_dict.get("roc_auc") is not None
                    else None
                ),
                confusion_matrix=metrics_dict.get("confusion_matrix"),
                custom_metrics=metrics_dict.get("custom_metrics", {}),
            )
        except (ValueError, TypeError) as e:
            raise MetricsParsingError(f"Invalid metric value: {e}")
        except ValidationError as e:
            raise MetricsParsingError(f"Validation error: {e}")

    @staticmethod
    def from_confusion_matrix(
        cm: Union[List[List[int]], np.ndarray], labels: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """
        Compute key metrics from a confusion matrix.

        Computes accuracy, precision, recall, and f1_score from a confusion matrix.
        Supports binary and multiclass classification.

        Args:
            cm: Confusion matrix as 2D list or numpy array
            labels: Optional list of class labels for reference

        Returns:
            EvaluationMetrics with computed values

        Raises:
            MetricsParsingError: If confusion matrix is invalid or malformed
        """
        try:
            if isinstance(cm, list):
                try:
                    cm = np.array(cm, dtype=float)
                except ValueError as e:
                    raise ValueError(
                        "Confusion matrix must be rectangular (jagged arrays not allowed)"
                    )
            elif not isinstance(cm, np.ndarray):
                raise TypeError("Confusion matrix must be a list or numpy array")

            if cm.ndim != 2:
                raise ValueError("Confusion matrix must be 2-dimensional")

            if cm.shape[0] != cm.shape[1]:
                raise ValueError(
                    "Confusion matrix must be square (n_classes x n_classes)"
                )

            cm = cm.astype(float)

            accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0

            per_class_recall = np.diag(cm) / np.sum(cm, axis=1)
            per_class_precision = np.diag(cm) / np.sum(cm, axis=0)

            with np.errstate(divide="ignore", invalid="ignore"):
                per_class_f1 = (
                    2
                    * (per_class_precision * per_class_recall)
                    / (per_class_precision + per_class_recall)
                )
                per_class_f1 = np.nan_to_num(per_class_f1)

            precision = float(np.mean(per_class_precision))
            recall = float(np.mean(per_class_recall))
            f1_score = float(np.mean(per_class_f1))

            return EvaluationMetrics(
                accuracy=float(accuracy),
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                roc_auc=None,
                confusion_matrix=cm.astype(int).tolist(),
                custom_metrics={"computed_from_confusion_matrix": True},
            )
        except (ValueError, TypeError, ZeroDivisionError) as e:
            raise MetricsParsingError(f"Invalid confusion matrix: {e}")
