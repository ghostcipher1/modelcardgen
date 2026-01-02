# CLEAR: Concise Logic and Explanation Analysis Reports

CLEAR is a Python tool for generating model cards and risk reports from machine learning evaluation outputs. It transforms structured evaluation data into readable, standardized documentation that can be shared with stakeholders.

## What This Tool Does

CLEAR takes ML evaluation metrics and metadata as input and generates:

- **Model Cards**: Standardized documentation including model overview, intended use, dataset summaries, performance metrics, limitations, and ethical considerations.
- **Risk Reports**: Analysis of identified risks, their mitigation strategies, and severity levels.
- **Markdown Output**: Human-readable markdown files suitable for documentation repositories or sharing with teams.

The tool processes JSON or YAML input files and applies templating to produce consistent, well-structured reports.

## What It Does NOT Do

CLEAR does not:

- Automatically compute ML metrics. You must provide pre-calculated evaluation results.
- Generate visualizations or charts. Output is text-based markdown only.
- Store or manage model artifacts, checkpoints, or weights.
- Connect to external APIs, cloud services, or model registries.
- Enforce particular ML frameworks or tooling choices.
- Make judgment calls about model safety or regulatory compliance. It documents what you provide.

## Installation

Install from PyPI:

```bash
pip install modelcardgen
```

Or install from source with development dependencies:

```bash
git clone https://github.com/ghostcipher1/modelcardgen.git
cd modelcardgen
pip install -e ".[dev]"
```

Requires Python 3.10 or later.

## CLI Usage Examples

### Generate a model card

```bash
modelcardgen generate --metrics evaluation.json --output-dir .
```

### Using YAML input

```bash
modelcardgen generate --metrics metrics.yaml --output-dir ./reports
```

### Validate metrics file

```bash
modelcardgen validate --metrics evaluation.json
```

### View help

```bash
modelcardgen --help
modelcardgen generate --help
```

## Input Data Schema

The tool expects JSON or YAML input files containing model metadata, dataset information, evaluation metrics, and risk assessments. Below is the complete input schema specification.

### Required Top-Level Fields

```yaml
model_name: string              # Name of the model
model_version: string           # Semantic version (e.g., "1.0.0")
model_description: string       # High-level overview
model_owner: string             # Person or team responsible
model_license: string           # License type (e.g., "Apache-2.0")
model_framework: string         # ML framework used (e.g., "scikit-learn")

accuracy: float                 # 0.0 to 1.0
precision: float                # 0.0 to 1.0
recall: float                   # 0.0 to 1.0
f1_score: float                 # 0.0 to 1.0
```

### Optional Fields

```yaml
model_release_date: YYYY-MM-DD  # Model release date (defaults to today)

roc_auc: float                  # 0.0 to 1.0 (optional)
confusion_matrix: [[int]]       # 2D array of prediction counts (optional)
custom_metrics: {}              # Dictionary of domain-specific metrics (optional)

training_data_name: string
training_data_description: string
training_data_size: integer     # Number of samples
training_data_features: [string]  # List of feature names
training_data_target: string    # Target variable name
training_data_source_url: url   # Optional URL to dataset source

eval_data_name: string
eval_data_description: string
eval_data_size: integer
eval_data_features: [string]
eval_data_target: string
eval_data_source_url: url

unsuitable_inputs: [string]     # List of input types where model fails
environmental_constraints: string  # Hardware/software requirements
out_of_scope_uses: [string]     # Scenarios to avoid

intended_users: [string]        # Target audience personas
intended_use_cases: [string]    # Specific tasks designed for
prohibited_uses: [string]       # Forbidden uses (ethical/legal)
```

### Risks Array (Optional)

```yaml
risks:
  - risk_type: string           # Category (e.g., "Data Bias")
    description: string         # Detailed explanation
    mitigation_strategy: string # Mitigation approach
    severity: string            # "Low", "Medium", or "High"
```

### Complete JSON Example

```json
{
  "model_name": "Email Spam Classifier",
  "model_version": "2.1.0",
  "model_description": "Classifies emails as spam or legitimate",
  "model_owner": "ML Team",
  "model_license": "Apache-2.0",
  "model_framework": "scikit-learn",
  "accuracy": 0.963,
  "precision": 0.951,
  "recall": 0.945,
  "f1_score": 0.948,
  "roc_auc": 0.985,
  "training_data_name": "Enron Email Corpus",
  "training_data_description": "Real email messages with labels",
  "training_data_size": 755000,
  "training_data_features": ["subject_line", "body_text"],
  "training_data_target": "spam_label",
  "eval_data_name": "Recent Email Dataset",
  "eval_data_description": "Holdout test set",
  "eval_data_size": 50000,
  "eval_data_features": ["subject_line", "body_text"],
  "eval_data_target": "spam_label",
  "unsuitable_inputs": ["Non-English emails", "Encrypted content"],
  "out_of_scope_uses": ["Real-time filtering without review"],
  "intended_users": ["Email administrators", "IT security teams"],
  "intended_use_cases": ["Spam detection"],
  "prohibited_uses": ["Discriminatory filtering"],
  "risks": [
    {
      "risk_type": "Data Distribution Shift",
      "description": "Production data may differ from training",
      "mitigation_strategy": "Monitor metrics in production",
      "severity": "Medium"
    }
  ]
}
```

### Complete YAML Example

```yaml
model_name: Email Spam Classifier
model_version: 2.1.0
model_description: Classifies emails as spam or legitimate
model_owner: ML Team
model_license: Apache-2.0
model_framework: scikit-learn

accuracy: 0.963
precision: 0.951
recall: 0.945
f1_score: 0.948
roc_auc: 0.985

training_data_name: Enron Email Corpus
training_data_description: Real email messages with labels
training_data_size: 755000
training_data_features:
  - subject_line
  - body_text
training_data_target: spam_label

eval_data_name: Recent Email Dataset
eval_data_description: Holdout test set
eval_data_size: 50000
eval_data_features:
  - subject_line
  - body_text
eval_data_target: spam_label

unsuitable_inputs:
  - Non-English emails
  - Encrypted content
out_of_scope_uses:
  - Real-time filtering without review

intended_users:
  - Email administrators
  - IT security teams
intended_use_cases:
  - Spam detection
prohibited_uses:
  - Discriminatory filtering

risks:
  - risk_type: Data Distribution Shift
    description: Production data may differ from training
    mitigation_strategy: Monitor metrics in production
    severity: Medium
```

### Validation Rules

- **Metrics values** (accuracy, precision, recall, f1_score, roc_auc) must be between 0.0 and 1.0
- **All model_* fields** are required
- **All training_data_* and eval_data_* fields are required** except source_url (optional)
- **All metrics fields** (accuracy, precision, recall, f1_score) are required; roc_auc is optional
- **Lists** (features, unsuitable_inputs, etc.) can be empty but must be arrays
- **Risks** is optional; if provided, each risk must have all four fields

### Common Errors

| Error | Solution |
|-------|----------|
| `Invalid JSON` | Check file syntax using `jq` or a JSON validator |
| `Invalid YAML` | Check indentation (use spaces, not tabs); use a YAML linter |
| `Validation failed: accuracy` | Ensure metric values are between 0.0 and 1.0 |
| `File not found` | Verify the file path and ensure the file exists |
| `Missing required field` | Check that all required model_* and eval_* fields are present |

## Python API Example

Use CLEAR as a library in your code:

```python
from modelcardgen.core.models import (
    ModelMetadata,
    DatasetMetadata,
    EvaluationMetrics,
    RiskAssessment,
)
from modelcardgen.reports.markdown import MarkdownCardGenerator

metadata = ModelMetadata(
    name="My Classifier",
    version="1.0.0",
    description="Classifies text documents.",
    owner="ML Team",
    license="Apache-2.0",
    framework="scikit-learn"
)

metrics = EvaluationMetrics(
    accuracy=0.92,
    precision=0.90,
    recall=0.94,
    f1_score=0.92,
    roc_auc=0.96
)

training_data = DatasetMetadata(
    name="Training Set",
    description="Internal labeled dataset",
    size=10000,
    features=["text_features"],
    target="label"
)

risks = [
    RiskAssessment(
        risk_type="Data Distribution Shift",
        description="Production data may differ from training distribution.",
        mitigation_strategy="Monitor performance metrics in production.",
        severity="Medium"
    )
]

generator = MarkdownCardGenerator()
generator.generate(
    metadata=metadata,
    metrics=metrics,
    training_data=training_data,
    risks=risks,
    output_path="MODEL_CARD.md"
)
```

## CI/CD Usage Example

Integrate model card generation into your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
name: Generate Model Card
on:
  push:
    paths:
      - 'model/evaluation_results.json'

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install modelcardgen
      - run: |
          modelcardgen generate \
            --input model/evaluation_results.json \
            --output docs/MODEL_CARD.md
      - run: git add docs/MODEL_CARD.md && git commit -m "Update model card"
        if: ${{ github.event_name == 'push' }}
```

## Design Philosophy

CLEAR follows these principles:

- **Offline First**: No external API calls or cloud dependencies. Everything runs locally.
- **Data Driven**: Accuracy depends on the quality of input data. Garbage in, garbage out.
- **Template Based**: Uses Jinja2 templating for flexibility. Customize output by modifying templates.
- **No Magic**: Explicit over implicit. The tool documents what you tell it; it doesn't infer or assume.
- **Minimal Dependencies**: Relies on standard, well-maintained Python libraries (Jinja2, Pydantic, Pandas).
- **Language Agnostic**: Works with any ML framework or language, as long as you can generate JSON/YAML evaluation output.

## License

Licensed under the Apache License 2.0. See `LICENSE` file for details.
