from datetime import date
from pathlib import Path
from typing import List, Optional, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateError

from modelcardgen.core.interpreter import MetricInterpreter
from modelcardgen.core.models import EvaluationMetrics, RiskAssessment

__all__ = [
    "RiskReportGenerator",
]


class RiskReportGenerator:
    """
    Generates a comprehensive Risk Report from RiskAssessment data.

    The report is conservative in scope and suitable for audit, compliance,
    and operational risk management. It emphasizes explicit "Do Not Use If"
    conditions and mandatory mitigation requirements.
    
    **API Stability**: Stable. Public API for risk report generation.
    The generate() method and risk classification logic are guaranteed stable.
    Report structure and severity classifications will remain consistent.
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
        model_name: str,
        model_version: str,
        risks: List[RiskAssessment],
        output_path: Union[str, Path] = "RISK_REPORT.md",
        report_date: Optional[date] = None,
        do_not_use_conditions: Optional[List[dict]] = None,
        metrics: Optional[EvaluationMetrics] = None,
        include_interpretations: bool = True,
    ) -> Path:
        """
        Generate a risk report Markdown file.

        Args:
            model_name: Name of the model being assessed.
            model_version: Semantic version of the model.
            risks: List of RiskAssessment objects identifying potential harms.
            output_path: Path where the risk report will be written.
            report_date: Date of the assessment (defaults to today).
            do_not_use_conditions: Optional list of dicts with 'condition' and 'explanation' keys.
                                  If not provided, defaults will be used.
            metrics: Optional EvaluationMetrics for metric-based risk analysis.
            include_interpretations: Whether to include metric interpretations. Defaults to True.

        Returns:
            Path to the generated risk report file.
            
        Raises:
            ValueError: If model_name, model_version, or risks are invalid.
            TemplateError: If template rendering fails.
            IOError: If output file cannot be written.
        """
        try:
            if not model_name or not isinstance(model_name, str):
                raise ValueError(f"model_name must be a non-empty string, got: {model_name}")
            
            if not model_version or not isinstance(model_version, str):
                raise ValueError(f"model_version must be a non-empty string, got: {model_version}")
            
            if not isinstance(risks, list):
                raise ValueError(f"risks must be a list, got: {type(risks).__name__}")
            
            if report_date is None:
                report_date = date.today()

            risk_summary = self._classify_risk_level(risks)

            interpretations = []
            if metrics and include_interpretations:
                interpretations = MetricInterpreter.interpret(metrics)

            template = self.env.get_template("risk_report.md.j2")

            context = {
                "model_name": model_name,
                "model_version": model_version,
                "risks": risks,
                "report_date": report_date,
                "risk_summary": risk_summary,
                "do_not_use_conditions": do_not_use_conditions or [],
                "interpretations": interpretations,
                "metrics_performance_summary": (
                    MetricInterpreter.get_summary_statement(metrics) if metrics else ""
                ),
            }

            content = template.render(**context)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")

            return output_path
        
        except ValueError as e:
            raise ValueError(f"Invalid input for risk report generation:\n  {str(e)}")
        except TemplateError as e:
            raise TemplateError(
                f"Failed to render risk report template:\n  {str(e)}"
            )
        except IOError as e:
            raise IOError(
                f"Failed to write risk report to {output_path}:\n  {str(e)}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during risk report generation:\n  {str(e)}"
            )

    @staticmethod
    def _classify_risk_level(risks: List[RiskAssessment]) -> str:
        """
        Classify overall risk level based on identified risks.

        Args:
            risks: List of RiskAssessment objects.

        Returns:
            A classification string suitable for the executive summary.
        """
        if not risks:
            return "minimal documented risk"

        severities = [risk.severity.lower() for risk in risks]

        if "high" in severities:
            return "significant risk"
        elif "medium" in severities:
            return "moderate risk"
        else:
            return "low documented risk"
