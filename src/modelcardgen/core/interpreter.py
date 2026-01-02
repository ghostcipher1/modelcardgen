from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from modelcardgen.core.models import EvaluationMetrics

__all__ = [
    "Interpretation",
    "MetricInterpreter",
]


class Interpretation(BaseModel):
    """
    A human-readable interpretation of model metrics.
    
    **API Stability**: Stable. Public API for metric interpretation results.
    """

    category: str = Field(
        ...,
        description="The aspect of the model being interpreted (e.g., 'recall', 'class_balance').",
    )
    severity: str = Field(
        ..., description="The importance level: 'low', 'medium', or 'high'."
    )
    statement: str = Field(
        ..., description="A human-readable finding about the metrics."
    )
    recommendation: str = Field(
        ..., description="Actionable advice based on the interpretation."
    )


class MetricInterpreter:
    """
    Rule-based interpreter that converts raw metrics into human-readable statements.

    Implements deterministic, documented rules without ML/LLM usage.
    Rules are organized by metric and include thresholds, logic, and explanations.
    
    **API Stability**: Stable. Public API for metric interpretation.
    The interpret() and get_summary_statement() methods are guaranteed stable.
    Internal rule implementations may improve but method signatures remain compatible.
    """

    @staticmethod
    def interpret(metrics: EvaluationMetrics) -> List[Interpretation]:
        """
        Analyze metrics and generate human-readable interpretations.

        Args:
            metrics: EvaluationMetrics object to interpret.

        Returns:
            List of Interpretation objects representing findings.
        """
        interpretations = []

        interpretations.extend(MetricInterpreter._interpret_recall(metrics))
        interpretations.extend(MetricInterpreter._interpret_precision(metrics))
        interpretations.extend(MetricInterpreter._interpret_class_balance(metrics))
        interpretations.extend(
            MetricInterpreter._interpret_overall_performance(metrics)
        )
        interpretations.extend(MetricInterpreter._interpret_roc_auc(metrics))

        return interpretations

    @staticmethod
    def _interpret_recall(metrics: EvaluationMetrics) -> List[Interpretation]:
        """
        Interpret recall metric.

        RULE: Recall measures the proportion of actual positive cases the model correctly identifies.
        - Recall < 0.60: HIGH severity - Model misses many positive cases (high false negatives).
        - Recall 0.60-0.75: MEDIUM severity - Model may miss notable portion of positive cases.
        - Recall 0.75-0.90: LOW severity - Model catches most positive cases with minor gaps.
        - Recall >= 0.90: No interpretation needed (excellent recall).

        Recommendation: When recall is low, investigate false negatives and consider:
        - Lowering decision threshold
        - Adjusting class weights during training
        - Collecting more positive examples
        """
        interpretations = []

        if metrics.recall < 0.60:
            interpretations.append(
                Interpretation(
                    category="recall",
                    severity="high",
                    statement=(
                        f"Recall is {metrics.recall:.1%}, indicating the model misses a significant portion "
                        f"of positive cases. For every 100 positive samples, the model identifies only ~{int(metrics.recall * 100)}."
                    ),
                    recommendation=(
                        "Investigate false negatives. Consider lowering the decision threshold, adjusting class weights, "
                        "or collecting additional positive training examples."
                    ),
                )
            )
        elif metrics.recall < 0.75:
            interpretations.append(
                Interpretation(
                    category="recall",
                    severity="medium",
                    statement=(
                        f"Recall is {metrics.recall:.1%}, meaning the model may miss a notable portion of positive cases. "
                        f"This could impact applications where missing positive cases has high cost."
                    ),
                    recommendation=(
                        "Evaluate whether the application can tolerate this miss rate. If not, optimize for recall "
                        "by adjusting the decision threshold or retraining with emphasis on positive case detection."
                    ),
                )
            )
        elif metrics.recall < 0.90:
            interpretations.append(
                Interpretation(
                    category="recall",
                    severity="low",
                    statement=(
                        f"Recall is {metrics.recall:.1%}, indicating the model catches most positive cases, "
                        f"though minor gaps remain (~{int((1 - metrics.recall) * 100)}% miss rate)."
                    ),
                    recommendation=(
                        "Performance is acceptable for most applications. Continue monitoring for improvement "
                        "in production if false negatives are costly."
                    ),
                )
            )

        return interpretations

    @staticmethod
    def _interpret_precision(metrics: EvaluationMetrics) -> List[Interpretation]:
        """
        Interpret precision metric.

        RULE: Precision measures the proportion of positive predictions that are correct.
        - Precision < 0.60: HIGH severity - Many false positives; model often makes incorrect positive predictions.
        - Precision 0.60-0.75: MEDIUM severity - Moderate false positive rate; may generate excessive alerts/actions.
        - Precision 0.75-0.90: LOW severity - Good precision with occasional false alarms.
        - Precision >= 0.90: No interpretation needed (excellent precision).

        Recommendation: When precision is low, the model makes many incorrect positive predictions.
        This is problematic when false positives are costly (e.g., fraud alerts, medical diagnoses).
        Consider: higher threshold, better features, or class rebalancing.
        """
        interpretations = []

        if metrics.precision < 0.60:
            interpretations.append(
                Interpretation(
                    category="precision",
                    severity="high",
                    statement=(
                        f"Precision is {metrics.precision:.1%}, indicating many false positives. "
                        f"Only ~{int(metrics.precision * 100)} of positive predictions are actually correct."
                    ),
                    recommendation=(
                        "High false positive rate may cause operational issues (alert fatigue, wasted resources). "
                        "Increase the decision threshold, improve features, or add negative example diversity to training data."
                    ),
                )
            )
        elif metrics.precision < 0.75:
            interpretations.append(
                Interpretation(
                    category="precision",
                    severity="medium",
                    statement=(
                        f"Precision is {metrics.precision:.1%}, meaning ~{int((1 - metrics.precision) * 100)}% of positive "
                        f"predictions are false positives. This may generate acceptable or unacceptable noise depending on use case."
                    ),
                    recommendation=(
                        "Evaluate operational impact of false positives. If costly, adjust decision threshold upward. "
                        "If acceptable, monitor in production."
                    ),
                )
            )
        elif metrics.precision < 0.90:
            interpretations.append(
                Interpretation(
                    category="precision",
                    severity="low",
                    statement=(
                        f"Precision is {metrics.precision:.1%}, indicating good accuracy of positive predictions "
                        f"with occasional false alarms."
                    ),
                    recommendation=(
                        "Performance is acceptable for most applications. Consider the trade-off between "
                        "false positives and false negatives for your use case."
                    ),
                )
            )

        return interpretations

    @staticmethod
    def _interpret_class_balance(metrics: EvaluationMetrics) -> List[Interpretation]:
        """
        Interpret class imbalance by analyzing accuracy vs. F1-score gap and confusion matrix.

        RULE: In balanced datasets, accuracy and F1-score are close. A large gap suggests class imbalance.
        - Gap > 0.15: HIGH severity - Accuracy inflated by majority class; metrics may be misleading.
        - Gap 0.10-0.15: MEDIUM severity - Some imbalance detected; F1 is more reliable than accuracy.
        - Gap < 0.10: LOW severity - Good balance between classes.

        Additionally, if confusion matrix is available, we check for extreme diagonal imbalance.

        Recommendation: When imbalance is detected, rely on F1, recall, and precision rather than accuracy.
        Consider resampling, class weights, or threshold adjustment.
        """
        interpretations = []

        accuracy_f1_gap = metrics.accuracy - metrics.f1_score

        if accuracy_f1_gap > 0.15:
            interpretations.append(
                Interpretation(
                    category="class_balance",
                    severity="high",
                    statement=(
                        f"Large gap between accuracy ({metrics.accuracy:.1%}) and F1-score ({metrics.f1_score:.1%}) "
                        f"suggests significant class imbalance. Accuracy is inflated by majority class performance."
                    ),
                    recommendation=(
                        "Do not rely on accuracy for evaluation. Use F1-score, precision, and recall instead. "
                        "Consider class weights, resampling, or threshold adjustment to address imbalance."
                    ),
                )
            )
        elif accuracy_f1_gap > 0.10:
            interpretations.append(
                Interpretation(
                    category="class_balance",
                    severity="medium",
                    statement=(
                        f"Moderate gap ({accuracy_f1_gap:.1%}) between accuracy and F1-score indicates some class imbalance. "
                        f"Accuracy may overstate true performance."
                    ),
                    recommendation=(
                        "Prioritize F1-score, recall, and precision in evaluation and reporting. "
                        "Explore class balancing techniques if the minority class is important."
                    ),
                )
            )
        elif accuracy_f1_gap < 0.10 and accuracy_f1_gap >= 0:
            interpretations.append(
                Interpretation(
                    category="class_balance",
                    severity="low",
                    statement=(
                        f"Small gap ({accuracy_f1_gap:.1%}) between accuracy and F1-score suggests "
                        f"reasonable class balance. Metrics are reliable."
                    ),
                    recommendation=(
                        "Classes appear balanced. Accuracy, precision, recall, and F1 are all meaningful metrics."
                    ),
                )
            )

        if metrics.confusion_matrix:
            interpretations.extend(
                MetricInterpreter._interpret_confusion_matrix(metrics.confusion_matrix)
            )

        return interpretations

    @staticmethod
    def _interpret_confusion_matrix(
        confusion_matrix: List[List[int]],
    ) -> List[Interpretation]:
        """
        Analyze confusion matrix for extreme class imbalance or prediction patterns.

        RULE: Extract total true positives, true negatives, false positives, and false negatives.
        For binary classification:
        - If true positives << true negatives: model is biased toward negative class.
        - If false positives >> true positives: model is too liberal (low threshold).
        - If false negatives >> false positives: model is too conservative (high threshold).
        """
        interpretations = []

        try:
            matrix = confusion_matrix

            if len(matrix) == 2:
                tn, fp = matrix[0][0], matrix[0][1]
                fn, tp = matrix[1][0], matrix[1][1]

                total_positive = tp + fn
                total_negative = tn + fp

                if total_positive > 0 and total_negative > 0:
                    imbalance_ratio = max(total_positive, total_negative) / min(
                        total_positive, total_negative
                    )

                    if imbalance_ratio > 10:
                        minority_class = (
                            "positive"
                            if total_positive < total_negative
                            else "negative"
                        )
                        interpretations.append(
                            Interpretation(
                                category="class_imbalance_severe",
                                severity="high",
                                statement=(
                                    f"Confusion matrix shows severe class imbalance (ratio {imbalance_ratio:.1f}:1). "
                                    f"The {minority_class} class is severely underrepresented in the dataset."
                                ),
                                recommendation=(
                                    "Apply stratified sampling, class weights, or synthetic data generation to rebalance training data. "
                                    "Ensure evaluation set proportions match production expectations."
                                ),
                            )
                        )
        except (IndexError, TypeError, ZeroDivisionError):
            pass

        return interpretations

    @staticmethod
    def _interpret_overall_performance(
        metrics: EvaluationMetrics,
    ) -> List[Interpretation]:
        """
        Interpret overall model performance quality.

        RULE: Combine accuracy, precision, recall, and F1 into performance tiers.
        - F1 >= 0.90 and accuracy >= 0.90: EXCELLENT - Ready for most production scenarios.
        - F1 >= 0.75: GOOD - Suitable for many applications with proper monitoring.
        - F1 >= 0.60: MODERATE - Limited applicability; understand risks carefully.
        - F1 < 0.60: POOR - Not ready for production without significant improvement.
        """
        interpretations = []

        f1 = metrics.f1_score

        if f1 >= 0.90 and metrics.accuracy >= 0.90:
            interpretations.append(
                Interpretation(
                    category="overall_performance",
                    severity="low",
                    statement=(
                        f"Overall performance is excellent (F1: {f1:.1%}, Accuracy: {metrics.accuracy:.1%}). "
                        f"Model demonstrates strong ability to identify both positive and negative cases accurately."
                    ),
                    recommendation=(
                        "Model is suitable for production deployment. Establish monitoring for performance drift "
                        "and continue regular evaluation cycles."
                    ),
                )
            )
        elif f1 >= 0.75:
            interpretations.append(
                Interpretation(
                    category="overall_performance",
                    severity="low",
                    statement=(
                        f"Overall performance is good (F1: {f1:.1%}). Model shows solid balance between precision and recall."
                    ),
                    recommendation=(
                        "Model is suitable for production with appropriate risk monitoring and operational controls. "
                        "Continue efforts to improve precision and recall."
                    ),
                )
            )
        elif f1 >= 0.60:
            interpretations.append(
                Interpretation(
                    category="overall_performance",
                    severity="medium",
                    statement=(
                        f"Overall performance is moderate (F1: {f1:.1%}). Model accuracy is limited; "
                        f"success depends heavily on use case requirements."
                    ),
                    recommendation=(
                        "Carefully evaluate whether this performance is acceptable for your use case. "
                        "Implement robust human-in-the-loop validation and comprehensive monitoring."
                    ),
                )
            )
        else:
            interpretations.append(
                Interpretation(
                    category="overall_performance",
                    severity="high",
                    statement=(
                        f"Overall performance is poor (F1: {f1:.1%}). Model is not ready for production deployment."
                    ),
                    recommendation=(
                        "Before deployment, significantly improve model performance through: better features, "
                        "more training data, hyperparameter tuning, or reconsidering the problem formulation."
                    ),
                )
            )

        return interpretations

    @staticmethod
    def _interpret_roc_auc(metrics: EvaluationMetrics) -> List[Interpretation]:
        """
        Interpret ROC AUC metric if available.

        RULE: ROC AUC measures the model's ability to distinguish between classes.
        - ROC AUC >= 0.90: EXCELLENT - Model has excellent discrimination ability.
        - ROC AUC >= 0.80: GOOD - Model has good discrimination ability.
        - ROC AUC >= 0.70: ACCEPTABLE - Model has fair discrimination ability.
        - ROC AUC < 0.70: POOR - Model discrimination is weak; may perform only slightly better than random.
        - ROC AUC = 0.50: Random guessing.
        """
        interpretations: List[Interpretation] = []

        if metrics.roc_auc is None:
            return interpretations

        roc_auc = metrics.roc_auc

        if roc_auc >= 0.90:
            interpretations.append(
                Interpretation(
                    category="roc_auc",
                    severity="low",
                    statement=(
                        f"ROC AUC is {roc_auc:.1%}, indicating excellent ability to distinguish between classes "
                        f"across all probability thresholds."
                    ),
                    recommendation=(
                        "Model demonstrates strong discrimination ability. Optimal threshold can be chosen "
                        "based on precision-recall trade-off for your application."
                    ),
                )
            )
        elif roc_auc >= 0.80:
            interpretations.append(
                Interpretation(
                    category="roc_auc",
                    severity="low",
                    statement=(
                        f"ROC AUC is {roc_auc:.1%}, indicating good discrimination ability between classes."
                    ),
                    recommendation=(
                        "Model is suitable for threshold-based classification. Consider operating point selection "
                        "based on precision-recall requirements."
                    ),
                )
            )
        elif roc_auc >= 0.70:
            interpretations.append(
                Interpretation(
                    category="roc_auc",
                    severity="medium",
                    statement=(
                        f"ROC AUC is {roc_auc:.1%}, indicating acceptable but not exceptional discrimination. "
                        f"Model performance varies significantly across different thresholds."
                    ),
                    recommendation=(
                        "Carefully select operating threshold based on precision-recall trade-off. "
                        "Consider ensemble methods or additional features to improve discrimination."
                    ),
                )
            )
        else:
            interpretations.append(
                Interpretation(
                    category="roc_auc",
                    severity="high",
                    statement=(
                        f"ROC AUC is {roc_auc:.1%}, indicating weak discrimination ability. "
                        f"Model performs only marginally better than random guessing."
                    ),
                    recommendation=(
                        "Model fundamentally lacks discrimination ability. Consider starting over with different features, "
                        "more data, or a different approach."
                    ),
                )
            )

        return interpretations

    @staticmethod
    def get_summary_statement(metrics: EvaluationMetrics) -> str:
        """
        Generate a concise one-sentence summary of model quality.

        Args:
            metrics: EvaluationMetrics object.

        Returns:
            A single summary statement.
        """
        f1 = metrics.f1_score

        if f1 >= 0.90:
            return "Model performance is excellent."
        elif f1 >= 0.75:
            return "Model performance is good."
        elif f1 >= 0.60:
            return "Model performance is moderate and requires careful evaluation."
        else:
            return "Model performance is poor and needs significant improvement."
