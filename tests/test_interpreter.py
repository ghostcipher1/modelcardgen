import pytest
from modelcardgen.core.interpreter import MetricInterpreter, Interpretation
from modelcardgen.core.models import EvaluationMetrics


@pytest.fixture
def metrics_excellent():
    return EvaluationMetrics(
        accuracy=0.95, precision=0.94, recall=0.96, f1_score=0.95, roc_auc=0.99
    )


@pytest.fixture
def metrics_good():
    return EvaluationMetrics(accuracy=0.82, precision=0.80, recall=0.85, f1_score=0.825)


@pytest.fixture
def metrics_moderate():
    return EvaluationMetrics(accuracy=0.70, precision=0.68, recall=0.72, f1_score=0.70)


@pytest.fixture
def metrics_poor():
    return EvaluationMetrics(accuracy=0.55, precision=0.50, recall=0.48, f1_score=0.49)


@pytest.fixture
def metrics_low_recall():
    return EvaluationMetrics(accuracy=0.75, precision=0.85, recall=0.50, f1_score=0.62)


@pytest.fixture
def metrics_low_precision():
    return EvaluationMetrics(accuracy=0.75, precision=0.55, recall=0.85, f1_score=0.67)


@pytest.fixture
def metrics_class_imbalanced():
    return EvaluationMetrics(
        accuracy=0.95,
        precision=0.92,
        recall=0.60,
        f1_score=0.73,
        confusion_matrix=[[950, 50], [40, 60]],
    )


@pytest.fixture
def metrics_severe_imbalance():
    return EvaluationMetrics(
        accuracy=0.98,
        precision=0.50,
        recall=0.30,
        f1_score=0.38,
        confusion_matrix=[[10000, 0], [700, 100]],
    )


class TestMetricInterpreterRecall:
    """Tests for recall metric interpretation."""

    def test_interpret_high_recall(self, metrics_excellent):
        interpretations = MetricInterpreter._interpret_recall(metrics_excellent)
        assert len(interpretations) == 0

    def test_interpret_low_recall_high_severity(self):
        metrics = EvaluationMetrics(
            accuracy=0.75, precision=0.85, recall=0.45, f1_score=0.60
        )
        interpretations = MetricInterpreter._interpret_recall(metrics)
        assert len(interpretations) == 1
        assert interpretations[0].category == "recall"
        assert interpretations[0].severity == "high"
        assert "misses" in interpretations[0].statement.lower()
        assert "45" in interpretations[0].statement

    def test_interpret_medium_recall(self):
        metrics = EvaluationMetrics(
            accuracy=0.78, precision=0.80, recall=0.70, f1_score=0.75
        )
        interpretations = MetricInterpreter._interpret_recall(metrics)
        assert len(interpretations) == 1
        assert interpretations[0].severity == "medium"
        assert "notable portion" in interpretations[0].statement.lower()

    def test_interpret_good_recall(self):
        metrics = EvaluationMetrics(
            accuracy=0.88, precision=0.87, recall=0.80, f1_score=0.84
        )
        interpretations = MetricInterpreter._interpret_recall(metrics)
        assert len(interpretations) == 1
        assert interpretations[0].severity == "low"
        assert "most positive cases" in interpretations[0].statement.lower()


class TestMetricInterpreterPrecision:
    """Tests for precision metric interpretation."""

    def test_interpret_high_precision(self, metrics_excellent):
        interpretations = MetricInterpreter._interpret_precision(metrics_excellent)
        assert len(interpretations) == 0

    def test_interpret_low_precision_high_severity(self):
        metrics = EvaluationMetrics(
            accuracy=0.75, precision=0.50, recall=0.85, f1_score=0.63
        )
        interpretations = MetricInterpreter._interpret_precision(metrics)
        assert len(interpretations) == 1
        assert interpretations[0].category == "precision"
        assert interpretations[0].severity == "high"
        assert "false positives" in interpretations[0].statement.lower()

    def test_interpret_medium_precision(self):
        metrics = EvaluationMetrics(
            accuracy=0.78, precision=0.70, recall=0.80, f1_score=0.75
        )
        interpretations = MetricInterpreter._interpret_precision(metrics)
        assert len(interpretations) == 1
        assert interpretations[0].severity == "medium"

    def test_interpret_good_precision(self):
        metrics = EvaluationMetrics(
            accuracy=0.88, precision=0.82, recall=0.86, f1_score=0.84
        )
        interpretations = MetricInterpreter._interpret_precision(metrics)
        assert len(interpretations) == 1
        assert interpretations[0].severity == "low"


class TestMetricInterpreterClassBalance:
    """Tests for class balance interpretation."""

    def test_interpret_balanced_classes(self):
        metrics = EvaluationMetrics(
            accuracy=0.85, precision=0.84, recall=0.86, f1_score=0.85
        )
        interpretations = MetricInterpreter._interpret_class_balance(metrics)
        assert len(interpretations) >= 1
        class_balance_interps = [
            i for i in interpretations if i.category == "class_balance"
        ]
        assert len(class_balance_interps) == 1
        assert class_balance_interps[0].severity == "low"
        assert "balance" in class_balance_interps[0].statement.lower()

    def test_interpret_moderate_imbalance(self):
        metrics = EvaluationMetrics(
            accuracy=0.92, precision=0.88, recall=0.75, f1_score=0.81
        )
        interpretations = MetricInterpreter._interpret_class_balance(metrics)
        class_balance_interps = [
            i for i in interpretations if i.category == "class_balance"
        ]
        assert len(class_balance_interps) == 1
        assert class_balance_interps[0].severity == "medium"

    def test_interpret_severe_imbalance(self, metrics_class_imbalanced):
        interpretations = MetricInterpreter._interpret_class_balance(
            metrics_class_imbalanced
        )
        class_balance_interps = [
            i for i in interpretations if i.category == "class_balance"
        ]
        assert len(class_balance_interps) == 1
        assert class_balance_interps[0].severity == "high"
        assert "large gap" in class_balance_interps[0].statement.lower()

    def test_interpret_confusion_matrix_severe_imbalance(
        self, metrics_severe_imbalance
    ):
        interpretations = MetricInterpreter._interpret_class_balance(
            metrics_severe_imbalance
        )
        severe_imbalance_interps = [
            i for i in interpretations if i.category == "class_imbalance_severe"
        ]
        assert len(severe_imbalance_interps) == 1
        assert severe_imbalance_interps[0].severity == "high"
        assert "severe" in severe_imbalance_interps[0].statement.lower()
        assert "12.5" in severe_imbalance_interps[0].statement


class TestMetricInterpreterOverallPerformance:
    """Tests for overall performance interpretation."""

    def test_interpret_excellent_performance(self, metrics_excellent):
        interpretations = MetricInterpreter._interpret_overall_performance(
            metrics_excellent
        )
        assert len(interpretations) == 1
        assert interpretations[0].category == "overall_performance"
        assert interpretations[0].severity == "low"
        assert "excellent" in interpretations[0].statement.lower()

    def test_interpret_good_performance(self, metrics_good):
        interpretations = MetricInterpreter._interpret_overall_performance(metrics_good)
        assert len(interpretations) == 1
        assert interpretations[0].severity == "low"
        assert "good" in interpretations[0].statement.lower()

    def test_interpret_moderate_performance(self, metrics_moderate):
        interpretations = MetricInterpreter._interpret_overall_performance(
            metrics_moderate
        )
        assert len(interpretations) == 1
        assert interpretations[0].severity == "medium"
        assert "moderate" in interpretations[0].statement.lower()

    def test_interpret_poor_performance(self, metrics_poor):
        interpretations = MetricInterpreter._interpret_overall_performance(metrics_poor)
        assert len(interpretations) == 1
        assert interpretations[0].severity == "high"
        assert "poor" in interpretations[0].statement.lower()
        assert "not ready" in interpretations[0].statement.lower()


class TestMetricInterpreterROCAUC:
    """Tests for ROC AUC metric interpretation."""

    def test_interpret_roc_auc_excellent(self, metrics_excellent):
        interpretations = MetricInterpreter._interpret_roc_auc(metrics_excellent)
        assert len(interpretations) == 1
        assert interpretations[0].category == "roc_auc"
        assert interpretations[0].severity == "low"
        assert "excellent" in interpretations[0].statement.lower()

    def test_interpret_roc_auc_good(self):
        metrics = EvaluationMetrics(
            accuracy=0.82, precision=0.80, recall=0.85, f1_score=0.825, roc_auc=0.85
        )
        interpretations = MetricInterpreter._interpret_roc_auc(metrics)
        assert len(interpretations) == 1
        assert interpretations[0].severity == "low"
        assert "good" in interpretations[0].statement.lower()

    def test_interpret_roc_auc_acceptable(self):
        metrics = EvaluationMetrics(
            accuracy=0.75, precision=0.73, recall=0.77, f1_score=0.75, roc_auc=0.75
        )
        interpretations = MetricInterpreter._interpret_roc_auc(metrics)
        assert len(interpretations) == 1
        assert interpretations[0].severity == "medium"
        assert "acceptable" in interpretations[0].statement.lower()

    def test_interpret_roc_auc_poor(self):
        metrics = EvaluationMetrics(
            accuracy=0.55, precision=0.52, recall=0.58, f1_score=0.55, roc_auc=0.60
        )
        interpretations = MetricInterpreter._interpret_roc_auc(metrics)
        assert len(interpretations) == 1
        assert interpretations[0].severity == "high"
        assert "weak" in interpretations[0].statement.lower()

    def test_interpret_roc_auc_none(self):
        metrics = EvaluationMetrics(
            accuracy=0.82, precision=0.80, recall=0.85, f1_score=0.825, roc_auc=None
        )
        interpretations = MetricInterpreter._interpret_roc_auc(metrics)
        assert len(interpretations) == 0


class TestMetricInterpreterIntegration:
    """Integration tests for the interpret method."""

    def test_interpret_returns_list(self, metrics_excellent):
        interpretations = MetricInterpreter.interpret(metrics_excellent)
        assert isinstance(interpretations, list)
        assert all(isinstance(i, Interpretation) for i in interpretations)

    def test_interpret_excellent_model(self, metrics_excellent):
        interpretations = MetricInterpreter.interpret(metrics_excellent)
        assert len(interpretations) >= 1

        categories = {i.category for i in interpretations}
        assert "overall_performance" in categories
        assert "roc_auc" in categories

    def test_interpret_poor_model(self, metrics_poor):
        interpretations = MetricInterpreter.interpret(metrics_poor)
        assert len(interpretations) >= 3

        severities = {i.severity for i in interpretations}
        assert "high" in severities

    def test_interpret_imbalanced_model(self, metrics_class_imbalanced):
        interpretations = MetricInterpreter.interpret(metrics_class_imbalanced)

        severity_high = [i for i in interpretations if i.severity == "high"]
        assert len(severity_high) >= 1

        categories = {i.category for i in interpretations}
        assert "class_balance" in categories or "recall" in categories

    def test_interpret_with_all_fields(self):
        metrics = EvaluationMetrics(
            accuracy=0.92,
            precision=0.90,
            recall=0.94,
            f1_score=0.92,
            roc_auc=0.96,
            confusion_matrix=[[450, 50], [30, 470]],
            custom_metrics={"specificity": 0.90},
        )
        interpretations = MetricInterpreter.interpret(metrics)

        assert len(interpretations) >= 2
        assert all(isinstance(i, Interpretation) for i in interpretations)

        for interp in interpretations:
            assert interp.category
            assert interp.severity in ["low", "medium", "high"]
            assert len(interp.statement) > 10
            assert len(interp.recommendation) > 10


class TestMetricInterpreterSummaryStatement:
    """Tests for the summary statement generation."""

    def test_summary_excellent(self, metrics_excellent):
        summary = MetricInterpreter.get_summary_statement(metrics_excellent)
        assert isinstance(summary, str)
        assert "excellent" in summary.lower()

    def test_summary_good(self, metrics_good):
        summary = MetricInterpreter.get_summary_statement(metrics_good)
        assert "good" in summary.lower()

    def test_summary_moderate(self, metrics_moderate):
        summary = MetricInterpreter.get_summary_statement(metrics_moderate)
        assert "moderate" in summary.lower()

    def test_summary_poor(self, metrics_poor):
        summary = MetricInterpreter.get_summary_statement(metrics_poor)
        assert "poor" in summary.lower()


class TestInterpretationModel:
    """Tests for the Interpretation Pydantic model."""

    def test_interpretation_creation(self):
        interp = Interpretation(
            category="test",
            severity="high",
            statement="Test statement",
            recommendation="Test recommendation",
        )
        assert interp.category == "test"
        assert interp.severity == "high"
        assert interp.statement == "Test statement"
        assert interp.recommendation == "Test recommendation"

    def test_interpretation_missing_required_field(self):
        with pytest.raises(ValueError):
            Interpretation(category="test", severity="high", statement="Test")


class TestConfusionMatrixEdgeCases:
    """Tests for confusion matrix parsing edge cases."""

    def test_confusion_matrix_binary_balanced(self):
        metrics = EvaluationMetrics(
            accuracy=0.80,
            precision=0.80,
            recall=0.80,
            f1_score=0.80,
            confusion_matrix=[[400, 100], [100, 400]],
        )
        interpretations = MetricInterpreter._interpret_confusion_matrix(
            metrics.confusion_matrix
        )
        assert len(interpretations) == 0

    def test_confusion_matrix_binary_slight_imbalance(self):
        metrics = EvaluationMetrics(
            accuracy=0.80,
            precision=0.80,
            recall=0.80,
            f1_score=0.80,
            confusion_matrix=[[900, 100], [100, 900]],
        )
        interpretations = MetricInterpreter._interpret_confusion_matrix(
            metrics.confusion_matrix
        )
        severe_imbalance = [
            i for i in interpretations if i.category == "class_imbalance_severe"
        ]
        assert len(severe_imbalance) == 0

    def test_confusion_matrix_extreme_imbalance(self):
        metrics = EvaluationMetrics(
            accuracy=0.99,
            precision=0.50,
            recall=0.05,
            f1_score=0.10,
            confusion_matrix=[[10000, 0], [500, 50]],
        )
        interpretations = MetricInterpreter._interpret_confusion_matrix(
            metrics.confusion_matrix
        )
        severe_imbalance = [
            i for i in interpretations if i.category == "class_imbalance_severe"
        ]
        assert len(severe_imbalance) == 1
        assert severe_imbalance[0].severity == "high"

    def test_confusion_matrix_invalid_input(self):
        interpretations = MetricInterpreter._interpret_confusion_matrix("invalid")
        assert len(interpretations) == 0

    def test_confusion_matrix_multiclass(self):
        multiclass_matrix = [[100, 10, 5], [5, 95, 10], [5, 5, 100]]
        interpretations = MetricInterpreter._interpret_confusion_matrix(
            multiclass_matrix
        )
        assert len(interpretations) == 0


class TestMetricInterpreterDeterminism:
    """Tests to ensure deterministic behavior."""

    def test_same_input_same_output(self):
        metrics = EvaluationMetrics(
            accuracy=0.80, precision=0.78, recall=0.82, f1_score=0.80
        )

        result1 = MetricInterpreter.interpret(metrics)
        result2 = MetricInterpreter.interpret(metrics)

        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            assert r1.category == r2.category
            assert r1.severity == r2.severity
            assert r1.statement == r2.statement
            assert r1.recommendation == r2.recommendation

    def test_interpretation_order_consistent(self):
        metrics = EvaluationMetrics(
            accuracy=0.60, precision=0.55, recall=0.50, f1_score=0.52
        )

        results = [MetricInterpreter.interpret(metrics) for _ in range(5)]

        for result in results[1:]:
            assert len(result) == len(results[0])
            for r, baseline in zip(result, results[0]):
                assert r.category == baseline.category
