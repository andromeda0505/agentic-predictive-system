from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)


# ============================================================
# Base strict model
# ============================================================

class StrictOutputModel(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )


# ============================================================
# Shared schemas
# ============================================================

class ValidationPlan(
    StrictOutputModel
):

    strategy: Literal[
        "temporal_holdout",
        "rolling_origin",
        "grouped_holdout",
        "random_holdout",
        "cross_validation",
        "not_yet_selected",
    ]

    # At this stage an exact cutoff has NOT
    # been deterministically selected.
    cutoff_defined: Literal[
        False
    ] = False

    cutoff: None = None

    rationale: str


class BaselineModel(
    StrictOutputModel
):

    name: str

    purpose: str


class CandidateModel(
    StrictOutputModel
):

    name: str

    family: Literal[
        "linear",
        "regularized_linear",
        "tree",
        "random_forest",
        "gradient_boosting",
        "survival",
        "other",
    ]

    status: Literal[
        "requires_empirical_comparison"
    ]

    rationale: str


class MetricSpec(
    StrictOutputModel
):

    name: str

    role: Literal[
        "primary",
        "secondary",
        "diagnostic",
    ]

    rationale: str


# ============================================================
# Planner output
# ============================================================

class PlannerOutput(
    StrictOutputModel
):

    target_construction: str

    data_quality_actions: list[str]

    feature_engineering: list[str]

    validation: ValidationPlan

    baselines: list[
        BaselineModel
    ]

    candidate_models: list[
        CandidateModel
    ]

    metrics: list[
        MetricSpec
    ]

    censoring_consideration: str

    unresolved_questions: list[str]

    next_step: str


# ============================================================
# Statistician output
# ============================================================

class CensoringAssessment(
    StrictOutputModel
):

    right_censoring_present: bool

    completed_case_regression_candidate: bool

    survival_analysis_candidate: bool

    note: str


class StatisticianOutput(
    StrictOutputModel
):

    sample_handling: list[str]

    missing_data_strategy: list[str]

    validation: ValidationPlan

    baselines: list[
        BaselineModel
    ]

    candidate_models: list[
        CandidateModel
    ]

    metrics: list[
        MetricSpec
    ]

    diagnostics: list[str]

    interactions_to_investigate: list[str]

    censoring: CensoringAssessment

    unresolved_questions: list[str]

    handoff: str

class CriticIssue(
    StrictOutputModel
):

    source_agent: Literal[
        "planner",
        "statistician",
    ]

    severity: Literal[
        "critical",
        "major",
        "minor",
    ]

    category: Literal[
        "target_definition",
        "column_hallucination",
        "prediction_time",
        "target_leakage",
        "unsupported_claim",
        "validation_design",
        "model_choice",
        "metric_choice",
        "missing_data",
        "censoring",
        "other",
    ]

    problematic_claim: str

    explanation: str

    recommended_fix: str


class CriticOutput(
    StrictOutputModel
):

    verdict: Literal[
        "pass",
        "revise",
    ]

    issues: list[
        CriticIssue
    ]

    strengths: list[str]

    recommended_next_action: Literal[
        "accept",
        "revise_planner",
        "revise_statistician",
        "revise_both",
    ]

    summary: str    