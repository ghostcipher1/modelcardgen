from datetime import date
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from modelcardgen.core.metrics import MetricsParser
from modelcardgen.core.models import (
    DatasetMetadata,
    EvaluationMetrics,
    ModelLimitations,
    ModelMetadata,
    RiskAssessment,
    UseCaseConstraints,
)

__all__ = [
    "ClassifierTrainer",
]


class ClassifierTrainer:
    """
    Convenience wrapper for training classifiers and extracting evaluation data.
    
    Handles train-test split, metric computation, and preparation of
    ModelMetadata, EvaluationMetrics, and other required objects.
    
    **API Stability**: Stable. Public convenience API for scikit-learn classifier training.
    The train_and_evaluate() and get_metrics() methods are guaranteed stable.
    """

    def __init__(self, model, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the trainer.

        Args:
            model: A fitted scikit-learn classifier with predict and predict_proba methods.
            test_size: Fraction of data to reserve for evaluation.
            random_state: Random seed for reproducibility.
        """
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.y_test = None
        self.y_pred = None
        self.metrics = None

    def train_and_evaluate(
        self, X_train, y_train, X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """
        Train the model and compute evaluation metrics.

        If X_test and y_test are not provided, they will be created from
        a train-test split of the provided X_train and y_train.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Optional test features. If None, created via train_test_split.
            y_test: Optional test labels. If None, created via train_test_split.

        Returns:
            EvaluationMetrics object with computed metrics.
        """
        if X_test is None or y_test is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train,
                y_train,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        self.model.fit(X_train, y_train)
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)

        report_dict = classification_report(
            y_test, self.y_pred, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, self.y_pred)

        self.metrics = MetricsParser.from_sklearn_report(report_dict)
        self.metrics.confusion_matrix = cm.tolist()

        return self.metrics

    def get_metrics(self) -> EvaluationMetrics:
        """
        Retrieve the computed evaluation metrics.

        Returns:
            EvaluationMetrics from the last train_and_evaluate call.

        Raises:
            ValueError: If train_and_evaluate has not been called yet.
        """
        if self.metrics is None:
            raise ValueError(
                "Metrics have not been computed. Call train_and_evaluate() first."
            )
        return self.metrics
