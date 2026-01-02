import json
from pathlib import Path

import click
import yaml

from modelcardgen import __version__
from modelcardgen.core.models import (DatasetMetadata, EvaluationMetrics,
                                      ModelLimitations, ModelMetadata,
                                      RiskAssessment, UseCaseConstraints)
from modelcardgen.core.security import (DEFAULT_MAX_YAML_SIZE,
                                        validate_yaml_file_size)
from modelcardgen.reports.markdown import MarkdownCardGenerator
from modelcardgen.reports.risk import RiskReportGenerator


@click.group()
@click.version_option(version=__version__)
def main():
    """CLEAR: Concise Logic and Explanation Analysis Reports.

    Generate model cards and risk reports from ML evaluation outputs.
    Model Cards document model metadata, performance, risks, and usage guidelines."""
    pass


def _load_file(file_path, max_yaml_size=DEFAULT_MAX_YAML_SIZE):
    """Load JSON or YAML file.

    Args:
        file_path: Path to JSON or YAML file
        max_yaml_size: Maximum allowed size for YAML files in bytes (default: 10 MB)

    Returns:
        Parsed data dictionary

    Raises:
        ValueError: If file format is not supported or exceeds size limits
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If JSON is malformed
        yaml.YAMLError: If YAML is malformed
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Please check the path and ensure the file exists."
        )

    if not file_path_obj.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".json"):
                return json.load(f)
            elif file_path.endswith((".yaml", ".yml")):
                validate_yaml_file_size(file_path, max_yaml_size)
                return yaml.safe_load(f)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path}\n"
                    f"Supported formats: .json, .yaml, .yml"
                )
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in {file_path}:\n"
            f"  Error at line {e.lineno}, column {e.colno}: {e.msg}\n"
            f"  Context: {e.doc[max(0, e.pos-40):e.pos+40]}",
            e.doc,
            e.pos,
        )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Invalid YAML in {file_path}:\n"
            f"  {str(e)}\n"
            f"  Check YAML syntax and indentation."
        )


def _build_model_objects(data, cm_data=None, allow_partial=False):
    """Build model objects from flattened data dictionary.

    Args:
        data: Input data dictionary with prefixed field names
        cm_data: Optional confusion matrix data
        allow_partial: If True, allow missing model metadata (only validate metrics)

    Returns:
        Tuple of (metadata, training_data, eval_data, metrics, limitations, constraints, risks)

    Raises:
        ValueError: If required fields are missing or invalid
        ValidationError: If field values don't match expected types/constraints
    """
    from pydantic import ValidationError

    try:
        model_fields = {k: v for k, v in data.items() if k.startswith("model_")}
        model_data = {k.replace("model_", ""): v for k, v in model_fields.items()}

        if allow_partial and not model_data:
            metadata = ModelMetadata(
                name="Unknown",
                version="0.0.0",
                description="",
                owner="",
                license="",
                framework="",
            )
        else:
            metadata = ModelMetadata(**model_data)

        train_fields = {k: v for k, v in data.items() if k.startswith("training_data_")}
        train_data = {
            k.replace("training_data_", ""): v for k, v in train_fields.items()
        }

        if allow_partial and not train_data:
            training_data = DatasetMetadata(name="Unknown", description="")
        else:
            training_data = DatasetMetadata(**train_data)

        eval_fields = {k: v for k, v in data.items() if k.startswith("eval_data_")}
        eval_data = {k.replace("eval_data_", ""): v for k, v in eval_fields.items()}

        if allow_partial and not eval_data:
            eval_dataset = DatasetMetadata(name="Unknown", description="")
        else:
            eval_dataset = DatasetMetadata(**eval_data)

        metrics_fields = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "confusion_matrix",
            "custom_metrics",
        ]
        metrics_data = {k: v for k, v in data.items() if k in metrics_fields}
        if cm_data:
            metrics_data.update(cm_data)
        metrics = EvaluationMetrics(**metrics_data)

        limit_fields = {
            k: v
            for k, v in data.items()
            if k
            in ["unsuitable_inputs", "out_of_scope_uses", "environmental_constraints"]
        }
        if not limit_fields:
            limit_fields = {"unsuitable_inputs": [], "out_of_scope_uses": []}
        limitations = ModelLimitations(**limit_fields)

        const_fields = {
            k: v
            for k, v in data.items()
            if k in ["intended_users", "intended_use_cases", "prohibited_uses"]
        }
        if not const_fields:
            const_fields = {
                "intended_users": [],
                "intended_use_cases": [],
                "prohibited_uses": [],
            }
        constraints = UseCaseConstraints(**const_fields)

        risks = [RiskAssessment(**risk) for risk in data.get("risks", [])]

        return (
            metadata,
            training_data,
            eval_dataset,
            metrics,
            limitations,
            constraints,
            risks,
        )

    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  - {field}: {msg}")

        raise ValueError(
            f"Validation failed for input data:\n"
            + "\n".join(errors)
            + f"\n\nCheck that all required fields are present and have correct types/values."
        )


@main.command()
@click.option(
    "--metrics",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Input metrics file (JSON or YAML) to validate.",
)
@click.option(
    "--confusion-matrix",
    type=click.Path(exists=True),
    default=None,
    help="Optional confusion matrix file (JSON).",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Enable strict validation mode (fail on low-quality metrics).",
)
def validate(metrics, confusion_matrix, strict):
    """Validate metrics file format and required fields."""
    from pydantic import ValidationError

    from modelcardgen.core.interpreter import MetricInterpreter

    try:
        click.echo(f"Validating metrics file {metrics}...")
        data = _load_file(metrics)

        cm_data = None
        if confusion_matrix:
            click.echo(f"Loading confusion matrix from {confusion_matrix}...")
            cm_data = _load_file(confusion_matrix)

        (
            metadata,
            training_data,
            eval_data,
            metrics_obj,
            limitations,
            constraints,
            risks,
        ) = _build_model_objects(data, cm_data, allow_partial=True)

        interpretations = MetricInterpreter.interpret(metrics_obj)
        has_high_severity = any(i.severity == "high" for i in interpretations)
        has_medium_severity = any(i.severity == "medium" for i in interpretations)

        if strict and (has_high_severity or has_medium_severity):
            click.echo("\nMetric Interpretation Analysis:")
            for interpretation in interpretations:
                click.echo(
                    f"  [{interpretation.severity.upper()}] {interpretation.category}: {interpretation.statement}"
                )
            click.echo(
                "\nValidation failed: Metrics do not meet strict quality criteria.",
                err=True,
            )
            raise SystemExit(2)
        elif has_high_severity or has_medium_severity:
            click.echo("\nMetric Interpretation Analysis:")
            for interpretation in interpretations:
                click.echo(
                    f"  [{interpretation.severity.upper()}] {interpretation.category}: {interpretation.statement}"
                )
            click.echo(
                "\nWarning: Validation passed but metrics show quality concerns.",
                err=True,
            )
            raise SystemExit(2)

        click.echo("Validation passed!")

    except FileNotFoundError as e:
        click.echo(f"Error: File not found\n  {str(e)}", err=True)
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON\n  {str(e)}", err=True)
        raise SystemExit(1)
    except yaml.YAMLError as e:
        click.echo(f"Error: Invalid YAML\n  {str(e)}", err=True)
        raise SystemExit(1)
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise SystemExit(1)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  - {field}: {msg}")
        error_msg = "\n".join(errors)
        click.echo(f"Error: Validation failed\n{error_msg}", err=True)
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"Error: An unexpected error occurred\n  {str(e)}", err=True)
        raise SystemExit(2)


@main.command()
@click.option(
    "--metrics",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Input metrics file (JSON or YAML).",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory for generated files (default: current directory).",
)
@click.option(
    "--confusion-matrix",
    type=click.Path(exists=True),
    default=None,
    help="Optional confusion matrix file (JSON).",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["md"]),
    default="md",
    help="Output format (default: md).",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Enable strict mode (warn on low-quality metrics).",
)
def generate(metrics, output_dir, confusion_matrix, format, strict):
    """Generate model cards and risk reports from evaluation metrics."""
    from pydantic import ValidationError

    try:
        click.echo(f"Reading metrics from {metrics}...")
        data = _load_file(metrics)

        cm_data = None
        if confusion_matrix:
            click.echo(f"Reading confusion matrix from {confusion_matrix}...")
            cm_data = _load_file(confusion_matrix)

        click.echo("Building model objects...")
        (
            metadata,
            training_data,
            eval_data,
            metrics_obj,
            limitations,
            constraints,
            risks,
        ) = _build_model_objects(data, cm_data)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        click.echo("Generating model card...")
        card_generator = MarkdownCardGenerator()
        card_path = card_generator.generate(
            metadata=metadata,
            training_data=training_data,
            eval_data=eval_data,
            metrics=metrics_obj,
            limitations=limitations,
            constraints=constraints,
            risks=risks,
            output_path=output_path / "MODEL_CARD.md",
        )

        click.echo("Generating risk report...")
        risk_generator = RiskReportGenerator()
        risk_path = risk_generator.generate(
            model_name=metadata.name,
            model_version=metadata.version,
            risks=risks,
            metrics=metrics_obj,
            output_path=output_path / "RISK_REPORT.md",
        )

        click.echo(f"Successfully generated model card: {card_path}")
        click.echo(f"Successfully generated risk report: {risk_path}")

    except FileNotFoundError as e:
        click.echo(f"Error: File not found\n  {str(e)}", err=True)
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON\n  {str(e)}", err=True)
        raise SystemExit(1)
    except yaml.YAMLError as e:
        click.echo(f"Error: Invalid YAML\n  {str(e)}", err=True)
        raise SystemExit(1)
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise SystemExit(2)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  - {field}: {msg}")
        error_msg = "\n".join(errors)
        click.echo(f"Error: Validation failed\n{error_msg}", err=True)
        raise SystemExit(2)
    except IOError as e:
        click.echo(f"Error: I/O error while writing output\n  {str(e)}", err=True)
        raise SystemExit(2)
    except Exception as e:
        click.echo(f"Error: An unexpected error occurred\n  {str(e)}", err=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
