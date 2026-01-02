from pathlib import Path
from typing import List, Optional, Union

from jinja2 import Environment, FileSystemLoader, TemplateError, select_autoescape

from modelcardgen.core.interpreter import MetricInterpreter
from modelcardgen.core.models import (
    DatasetMetadata,
    EvaluationMetrics,
    ModelCard,
    ModelLimitations,
    ModelMetadata,
    RiskAssessment,
    UseCaseConstraints,
)

__all__ = [
    "MarkdownCardGenerator",
]


class MarkdownCardGenerator:
    """
    Generates a human-readable Markdown model card from structured model information.

    Renders ModelMetadata, DatasetMetadata, EvaluationMetrics, and other model
    information into a comprehensive MODEL_CARD.md file using Jinja2 templates.

    **API Stability**: Stable. Public API for model card generation.
    The generate() and generate_from_model_card() methods are guaranteed stable.
    Output format may be enhanced but will remain backward compatible.
    """

    def __init__(self):
        """Initialize the template environment."""
        template_dir = Path(__file__).parent.parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(enabled_extensions=("jinja2",)),
        )

    def generate(
        self,
        metadata: ModelMetadata,
        training_data: DatasetMetadata,
        eval_data: DatasetMetadata,
        metrics: EvaluationMetrics,
        limitations: ModelLimitations,
        constraints: UseCaseConstraints,
        risks: Optional[List[RiskAssessment]] = None,
        output_path: Union[str, Path] = "MODEL_CARD.md",
        include_interpretations: bool = True,
    ) -> Path:
        """
        Generate a model card Markdown file.

        Args:
            metadata: Model metadata and versioning information.
            training_data: Information about the training dataset.
            eval_data: Information about the evaluation dataset.
            metrics: Quantitative performance measurements.
            limitations: Technical and contextual model boundaries.
            constraints: Policy and operational requirements.
            risks: Optional list of RiskAssessment items. Defaults to empty list.
            output_path: Path where the model card will be written.
            include_interpretations: Whether to include metric interpretations. Defaults to True.

        Returns:
            Path to the generated model card file.

        Raises:
            TemplateError: If template rendering fails.
            IOError: If output file cannot be written.
            ValueError: If required input parameters are invalid.
        """
        try:
            if risks is None:
                risks = []

            interpretations = []
            if include_interpretations:
                interpretations = MetricInterpreter.interpret(metrics)

            template = self.env.get_template("model_card.md.j2")

            context = {
                "metadata": metadata,
                "training_data": training_data,
                "eval_data": eval_data,
                "metrics": metrics,
                "limitations": limitations,
                "constraints": constraints,
                "risks": risks,
                "interpretations": interpretations,
                "performance_summary": MetricInterpreter.get_summary_statement(metrics),
            }

            content = template.render(**context)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")

            return output_path

        except TemplateError as e:
            raise TemplateError(f"Failed to render model card template:\n  {str(e)}")
        except IOError as e:
            raise IOError(f"Failed to write model card to {output_path}:\n  {str(e)}")
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during model card generation:\n  {str(e)}"
            )

    def generate_from_model_card(
        self, model_card: ModelCard, output_path: Union[str, Path] = "MODEL_CARD.md"
    ) -> Path:
        """
        Generate a model card from a complete ModelCard object.

        This is a convenience method that accepts a ModelCard instance
        and extracts the necessary components for report generation.

        Args:
            model_card: Complete ModelCard instance containing all required data.
            output_path: Path where the model card will be written.

        Returns:
            Path to the generated model card file.
        """
        return self.generate(
            metadata=model_card.metadata,
            training_data=model_card.training_data,
            eval_data=model_card.eval_data,
            metrics=model_card.metrics,
            limitations=model_card.limitations,
            constraints=model_card.constraints,
            risks=model_card.risks,
            output_path=output_path,
        )
