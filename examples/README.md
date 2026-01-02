# CLEAR Examples

This directory contains example input files, sample data, and scripts for the CLEAR (Concise Logic and Explanation Analysis Reports) model card and risk report generation tool.

## Contents Overview

| File | Purpose | Format | Use Case |
|------|---------|--------|----------|
| `input_minimal.json` | Python API minimal example | Nested JSON | Direct Python usage |
| `input_minimal_flat.json` | CLI minimal example | Flat JSON | CLI generation |
| `input_full.yaml` | Python API full example | Nested YAML | Python with comments |
| `input_full_flat.yaml` | CLI full example | Flat YAML | CLI with documentation |
| `generate_sample_data.py` | Data generation script | Python | Create evaluation datasets |
| `sample_email_evaluation_100k.csv` | Sample data (CSV) | CSV | Data analysis, testing |
| `sample_email_evaluation_100k.parquet` | Sample data (Parquet) | Parquet | Cloud storage, S3 upload |

## Input Configuration Files

### Format: Nested vs. Flat

CLEAR supports **two input formats**:

#### Nested Format (Python API)
Use for direct Python usage with the library classes.

**Files**: `input_minimal.json`, `input_full.yaml`

**Example structure**:
```json
{
  "metadata": {
    "name": "Model Name",
    "version": "1.0.0"
  },
  "training_data": {...},
  "metrics": {...}
}
```

**Load in Python**:
```python
from modelcardgen.reports.markdown import MarkdownCardGenerator
import json

with open('input_minimal.json') as f:
    data = json.load(f)

generator = MarkdownCardGenerator()
generator.generate(
    metadata=data['metadata'],
    training_data=data['training_data'],
    ...
)
```

#### Flat Format (CLI)
Use for command-line generation with flattened field names.

**Files**: `input_minimal_flat.json`, `input_full_flat.yaml`

**Example structure**:
```yaml
model_name: "Model Name"
model_version: "1.0.0"
model_owner: "Team"
training_data_name: "Dataset"
...
```

**Generate from command line**:
```bash
modelcardgen generate --metrics input_minimal_flat.json --output-dir output/
modelcardgen generate --metrics input_full_flat.yaml --output-dir output/
```

### Nested Format Examples

#### `input_minimal.json`
**Minimal example** demonstrating required fields for model card generation.

- **Purpose**: Quick reference for basic model card structure
- **Use Case**: Getting started with minimal information  
- **Features**: Basic binary classification model
- **Metrics**: Accuracy, precision, recall, F1-score, ROC AUC
- **Risks**: 2 simple risk assessments

**Use with Python API**:
```python
import json
from modelcardgen.reports.markdown import MarkdownCardGenerator

with open('examples/input_minimal.json') as f:
    data = json.load(f)

generator = MarkdownCardGenerator()
generator.generate(
    metadata=data['metadata'],
    training_data=data['training_data'],
    eval_data=data['eval_data'],
    metrics=data['metrics'],
    limitations=data['limitations'],
    constraints=data['constraints'],
    risks=data['risks'],
    output_path='MODEL_CARD.md'
)
```

#### `input_full.yaml`
**Comprehensive example** with all available fields and detailed annotations.

- **Purpose**: Full reference demonstrating all capabilities
- **Use Case**: Email spam classifier with real-world complexity
- **Features**: Complete model metadata, dataset information, comprehensive risk assessments
- **Documentation**: Inline comments explaining each field
- **Risks**: 5 detailed risk assessments (data bias, drift, adversarial robustness, etc.)

**Use with Python API**:
```python
import yaml
from modelcardgen.reports.markdown import MarkdownCardGenerator

with open('examples/input_full.yaml') as f:
    data = yaml.safe_load(f)

generator = MarkdownCardGenerator()
generator.generate(
    metadata=data['metadata'],
    training_data=data['training_data'],
    eval_data=data['eval_data'],
    metrics=data['metrics'],
    limitations=data['limitations'],
    constraints=data['constraints'],
    risks=data['risks'],
    output_path='MODEL_CARD.md'
)
```

### Flat Format Examples (CLI)

#### `input_minimal_flat.json`
**CLI-compatible minimal example** with all required fields.

- **Format**: Flattened field names with `model_`, `training_data_`, `eval_data_` prefixes
- **Use**: Command-line generation without Python scripting
- **Compatible with**: `modelcardgen generate --metrics`

**Usage**:
```bash
modelcardgen generate --metrics examples/input_minimal_flat.json --output-dir ./output
```

#### `input_full_flat.yaml`
**CLI-compatible comprehensive example** with detailed documentation.

- **Format**: Flattened structure with inline comments
- **Use**: Production-ready CLI generation with full documentation
- **Features**: Email spam classifier, 5 risks, comprehensive constraints

**Usage**:
```bash
modelcardgen generate --metrics examples/input_full_flat.yaml --output-dir ./output
```

### Sample Data

#### `sample_email_evaluation_100k.csv`
Large evaluation dataset with 100,000 email samples in CSV format.

- **Size**: ~16 MB
- **Rows**: 100,000 email records
- **Features**: 23 columns including:
  - Email metadata (sender domain, recipient count, attachments)
  - Content features (subject/body length, HTML content, links)
  - Authentication (SPF, DKIM, DMARC validation)
  - Model predictions and confidence scores
  - Ground truth spam labels

**Use Case**: Testing, benchmarking, and analysis of model evaluation data.

**Load in Python:**
```python
import pandas as pd
df = pd.read_csv('sample_email_evaluation_100k.csv')
print(df.shape)  # (100000, 23)
```

#### `sample_email_evaluation_100k.parquet`
Same 100k email dataset in Apache Parquet format (optimized for cloud storage).

- **Size**: ~3.7 MB (23% of CSV size with compression)
- **Format**: Snappy compression
- **Advantage**: More efficient for cloud storage (S3, Azure Blob, GCS)

**Use Case**: Production data pipelines, S3 uploads, columnar analytics.

**Load in Python:**
```python
import pandas as pd
df = pd.read_parquet('sample_email_evaluation_100k.parquet')
print(df.shape)  # (100000, 23)
```

**Upload to AWS S3:**
```bash
aws s3 cp sample_email_evaluation_100k.parquet s3://your-bucket/data/
```

### Data Generation Script

#### `generate_sample_data.py`
Python script that generates synthetic email evaluation datasets.

**Features**:
- Configurable number of samples (default: 100,000)
- Realistic feature distributions (email metadata, authentication, content)
- Probabilistic spam labels based on feature combinations
- Model predictions with confidence scores
- Outputs both CSV and Parquet formats

**Usage:**
```bash
# Generate default 100k samples
python generate_sample_data.py

# Or modify and run with custom sample count
# Edit num_samples parameter in generate_sample_email_data()
python generate_sample_data.py
```

**Output**:
- `sample_email_evaluation_100k.csv` (~16 MB)
- `sample_email_evaluation_100k.parquet` (~3.7 MB)

**Dependencies**:
- pandas
- numpy
- pyarrow (for parquet support)

## Field Reference

### Minimal Required Fields

Every model card must include:

```json
{
  "metadata": {
    "name": "Model Name",
    "version": "1.0.0",
    "description": "...",
    "owner": "Team",
    "license": "MIT",
    "framework": "scikit-learn"
  },
  "training_data": {...},
  "eval_data": {...},
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.85,
    "f1_score": 0.87
  },
  "limitations": {...},
  "constraints": {...},
  "risks": [...]
}
```

### Optional Fields

- `training_data.source_url` - URL to dataset source
- `metrics.roc_auc` - Area under ROC curve
- `metrics.confusion_matrix` - 2D prediction counts
- `metrics.custom_metrics` - Domain-specific metrics
- `limitations.environmental_constraints` - Hardware/software requirements

## Workflow Examples

### Example 1: CLI - Generate from Minimal Flat JSON
```bash
modelcardgen generate --metrics examples/input_minimal_flat.json --output-dir ./output
```

Outputs: `./output/MODEL_CARD.md` and `./output/RISK_REPORT.md`

### Example 2: CLI - Generate from Full Flat YAML
```bash
modelcardgen generate --metrics examples/input_full_flat.yaml --output-dir ./output
```

### Example 3: Python API - Generate from Nested JSON
```python
import json
from modelcardgen.reports.markdown import MarkdownCardGenerator
from modelcardgen.reports.risk import RiskReportGenerator

# Load nested format data
with open('examples/input_minimal.json') as f:
    data = json.load(f)

# Generate model card
card_gen = MarkdownCardGenerator()
card_gen.generate(
    metadata=data['metadata'],
    training_data=data['training_data'],
    eval_data=data['eval_data'],
    metrics=data['metrics'],
    limitations=data['limitations'],
    constraints=data['constraints'],
    risks=data['risks'],
    output_path='MODEL_CARD.md'
)

# Generate risk report
risk_gen = RiskReportGenerator()
risk_gen.generate(
    model_name=data['metadata']['name'],
    model_version=data['metadata']['version'],
    risks=data['risks'],
    output_path='RISK_REPORT.md'
)
```

### Example 4: Python API - Generate from Nested YAML
```python
import yaml
from modelcardgen.reports.markdown import MarkdownCardGenerator

with open('examples/input_full.yaml') as f:
    data = yaml.safe_load(f)

generator = MarkdownCardGenerator()
generator.generate(
    metadata=data['metadata'],
    training_data=data['training_data'],
    eval_data=data['eval_data'],
    metrics=data['metrics'],
    limitations=data['limitations'],
    constraints=data['constraints'],
    risks=data['risks'],
    output_path='MODEL_CARD_FULL.md'
)
```

### Example 5: Analyze Parquet Data and Generate Model Card
```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modelcardgen.core.models import (
    ModelMetadata, DatasetMetadata, EvaluationMetrics,
    ModelLimitations, UseCaseConstraints
)
from modelcardgen.reports.markdown import MarkdownCardGenerator

# Load evaluation data
df = pd.read_parquet('examples/sample_email_evaluation_100k.parquet')

# Calculate metrics from evaluation set
metrics = EvaluationMetrics(
    accuracy=accuracy_score(df['spam_label'], df['predicted_label']),
    precision=precision_score(df['spam_label'], df['predicted_label']),
    recall=recall_score(df['spam_label'], df['predicted_label']),
    f1_score=f1_score(df['spam_label'], df['predicted_label'])
)

print(f"Model Accuracy: {metrics.accuracy:.3f}")

# Create model card
generator = MarkdownCardGenerator()
generator.generate(
    metadata=ModelMetadata(
        name="Spam Classifier",
        version="1.0.0",
        description="Trained on email evaluation data",
        owner="ML Team",
        license="MIT",
        framework="scikit-learn"
    ),
    training_data=DatasetMetadata(
        name="Training Set",
        description="Historical emails",
        size=75000
    ),
    eval_data=DatasetMetadata(
        name="Evaluation Set",
        description="Recent emails",
        size=len(df)
    ),
    metrics=metrics,
    limitations=ModelLimitations(
        unsuitable_inputs=["Non-email text"],
        out_of_scope_uses=["Production without review"]
    ),
    constraints=UseCaseConstraints(
        intended_users=["Security team"],
        intended_use_cases=["Spam filtering"],
        prohibited_uses=["Blocking user mail without human review"]
    ),
    risks=[],
    output_path='MODEL_CARD.md'
)
```

### Example 6: Validate Metrics Before Generation
```bash
# Validate CLI format
modelcardgen validate --metrics examples/input_minimal_flat.json

# Validate with strict mode
modelcardgen validate --metrics examples/input_full_flat.yaml --strict
```

## Sample Data Statistics

### Email Evaluation Dataset (100k rows)

| Metric | Value |
|--------|-------|
| Total Records | 100,000 |
| Spam Ratio | 23.85% |
| Model Accuracy | 45.56% |
| Average Recipients | 5.73 |
| Attachments Per Email | 0.46 |
| Features | 23 columns |

### Column Details

**Identifiers & Timestamps**:
- `email_id` - Unique email identifier
- `timestamp` - Email timestamp (ISO format)

**Sender Information**:
- `sender_domain` - Domain of email sender
- `sender_reputation_score` - 0.0-1.0 reputation score
- `domain_age_days` - Age of sender domain in days

**Email Characteristics**:
- `recipient_count` - Number of recipients
- `attachment_count` - Number of attachments
- `subject_length` - Length of subject line
- `body_length` - Length of email body
- `html_content` - Boolean: contains HTML
- `contains_links` - Boolean: contains hyperlinks
- `contains_suspicious_keywords` - Boolean: suspicious content

**Authentication**:
- `spf_valid` - SPF validation passed
- `dkim_valid` - DKIM validation passed
- `dmarc_policy` - DMARC policy (none, quarantine, reject, missing)
- `from_header_matches_spf` - SPF alignment check

**Account Information**:
- `reply_to_count` - Number of reply-to addresses
- `previous_contact` - Boolean: previous contact with sender
- `account_age_days` - Age of recipient account in days

**Model Output**:
- `model_confidence` - 0.0-1.0 confidence score
- `spam_label` - Ground truth label (0=legitimate, 1=spam)
- `predicted_label` - Model prediction (0 or 1)
- `prediction_correct` - Boolean: prediction matches ground truth

## Integration Guide

### Using Examples in Your Project

1. **Start with minimal.json** to understand basic structure
2. **Review full.yaml** for complete field documentation
3. **Use sample data** for testing and validation
4. **Run generate script** to create larger datasets

### Input File Validation

```bash
# Validate with strict mode
modelcardgen validate input_full.yaml --strict

# Validate all fields are present
modelcardgen validate input_minimal.json
```

### Format Conversion

Convert between JSON and YAML:

```python
import json
import yaml

# JSON to YAML
with open('input_minimal.json') as f:
    data = json.load(f)
with open('input_minimal.yaml', 'w') as f:
    yaml.dump(data, f)

# YAML to JSON
with open('input_full.yaml') as f:
    data = yaml.safe_load(f)
with open('input_full.json', 'w') as f:
    json.dump(data, f, indent=2)
```

## Tips & Best Practices

1. **Use YAML for human readability** - More readable with comments and multiline strings
2. **Use JSON for programmatic generation** - Easier to generate from scripts
3. **Validate before generating** - Use `modelcardgen validate` first
4. **Start minimal** - Add fields incrementally as needed
5. **Keep parquet for production** - Better compression and query performance for large datasets
6. **Document risks thoroughly** - Include specific severity levels and mitigation strategies

## See Also

- [CLEAR Documentation](../README.md)
- [Model Card Best Practices](https://arxiv.org/abs/1810.03993)
- [Risk Assessment Framework](../EXAMPLE_RISK_REPORT.md)
- [Project Repository](https://github.com/ghostcipher1/modelcardgen)
