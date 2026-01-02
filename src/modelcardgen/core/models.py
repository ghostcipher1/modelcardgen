from datetime import date
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, field_validator

from modelcardgen.core.security import validate_url

__all__ = [
    "ModelMetadata",
    "DatasetMetadata",
    "EvaluationMetrics",
    "RiskAssessment",
    "ModelLimitations",
    "UseCaseConstraints",
    "ModelCard",
]


class ModelMetadata(BaseModel):
    """
    Basic descriptive information about the machine learning model.

    **API Stability**: Stable. This is a core public API class using Pydantic v2 for validation.
    Field names and types are guaranteed to remain compatible within v0.x versions.
    """

    name: str = Field(..., description="The official name of the model.")
    version: str = Field(..., description="The semantic version of the model.")
    description: str = Field(
        ..., description="A high-level overview of what the model does."
    )
    owner: str = Field(..., description="The person or team responsible for the model.")
    license: str = Field(..., description="The license governing the model's use.")
    release_date: date = Field(
        default_factory=date.today, description="The date the model was released."
    )
    framework: str = Field(
        ..., description="The technical framework used (e.g., PyTorch, Scikit-learn)."
    )


class DatasetMetadata(BaseModel):
    """
    Information about the data used to train or evaluate the model.

    **API Stability**: Stable. Public API for dataset documentation.
    """

    name: str = Field(..., description="The name of the dataset.")
    description: str = Field(
        ..., description="A summary of the dataset's contents and purpose."
    )
    source_url: Optional[HttpUrl] = Field(
        None, description="A link to the dataset's origin."
    )
    size: Optional[int] = Field(
        None, description="The number of samples in the dataset."
    )
    features: List[str] = Field(
        default_factory=list, description="A list of input feature names."
    )
    target: Optional[str] = Field(None, description="The name of the target variable.")

    @field_validator("source_url", mode="after")
    @classmethod
    def validate_source_url(cls, v: Optional[HttpUrl]) -> Optional[HttpUrl]:
        if v is None:
            return v
        validate_url(str(v))
        return v


class EvaluationMetrics(BaseModel):
    """
    Quantitative performance measurements for classification models.

    **API Stability**: Stable. Core metrics schema for model evaluation.
    All standard fields (accuracy, precision, recall, f1_score) are guaranteed stable.
    """

    accuracy: float = Field(
        ..., ge=0, le=1, description="The overall ratio of correct predictions."
    )
    precision: float = Field(
        ...,
        ge=0,
        le=1,
        description="The ability of the classifier not to label a negative sample as positive.",
    )
    recall: float = Field(
        ...,
        ge=0,
        le=1,
        description="The ability of the classifier to find all the positive samples.",
    )
    f1_score: float = Field(
        ..., ge=0, le=1, description="The harmonic mean of precision and recall."
    )
    roc_auc: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Area under the receiver operating characteristic curve.",
    )
    confusion_matrix: Optional[List[List[int]]] = Field(
        None, description="A 2D array representing prediction counts across classes."
    )
    custom_metrics: Dict[str, Union[float, str]] = Field(
        default_factory=dict, description="Any additional domain-specific metrics."
    )


class RiskAssessment(BaseModel):
    """
    Identification and mitigation of potential harms.

    **API Stability**: Stable. Public API for risk documentation and reporting.
    """

    risk_type: str = Field(
        ..., description="The category of risk (e.g., Bias, Security, Robustness)."
    )
    description: str = Field(
        ..., description="Detailed explanation of the specific risk identified."
    )
    mitigation_strategy: str = Field(
        ..., description="Steps taken or recommended to reduce this risk."
    )
    severity: str = Field(
        ...,
        description="The impact level if the risk occurs (e.g., Low, Medium, High).",
    )


class ModelLimitations(BaseModel):
    """
    Technical and contextual boundaries of the model's performance.

    **API Stability**: Stable. Public API for documenting model constraints.
    """

    unsuitable_inputs: List[str] = Field(
        ..., description="Input types or values where the model is known to fail."
    )
    environmental_constraints: Optional[str] = Field(
        None, description="Hardware or software requirements for deployment."
    )
    out_of_scope_uses: List[str] = Field(
        ..., description="Scenarios where the model should not be applied."
    )


class UseCaseConstraints(BaseModel):
    """
    Policy and operational requirements for applying the model.

    **API Stability**: Stable. Public API for use case documentation.
    """

    intended_users: List[str] = Field(
        ..., description="The primary audience or personas for the model outputs."
    )
    intended_use_cases: List[str] = Field(
        ..., description="The specific tasks the model was designed to perform."
    )
    prohibited_uses: List[str] = Field(
        ...,
        description="Uses that are strictly forbidden due to ethical or legal reasons.",
    )


class ModelCard(BaseModel):
    """
    The complete internal representation of a Model Card.

    **API Stability**: Stable. Top-level container for all model documentation.
    All component classes are stable Pydantic v2 models.
    """

    metadata: ModelMetadata
    training_data: DatasetMetadata
    eval_data: DatasetMetadata
    metrics: EvaluationMetrics
    risks: List[RiskAssessment]
    limitations: ModelLimitations
    constraints: UseCaseConstraints
