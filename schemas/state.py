from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
)


class AgentArtifact(
    BaseModel
):

    agent: str

    skill_version: str

    content: str

    schema_name: (
        str
        |
        None
    ) = None

    structured_data: (
        dict[str, Any]
        |
        None
    ) = None


class FeatureRule(
    BaseModel
):

    decision: Literal[
        "allow",
        "block",
        "verify",
        "identifier",
        "reference_time",
        "training_only",
    ]

    reason: str


class ToolExecution(
    BaseModel
):

    node: str

    tool: str

    arguments: dict[
        str,
        Any,
    ]

    result: str

    success: bool


class WorkflowState(
    BaseModel
):

    task: str

    dataset_path: str

    dataset_facts: dict[
        str,
        Any,
    ]

    feature_policy: dict[
        str,
        FeatureRule,
    ]

    artifacts: dict[
        str,
        AgentArtifact,
    ] = Field(
        default_factory=dict
    )

    warnings: list[
        str
    ] = Field(
        default_factory=list
    )

    tool_log: list[
        ToolExecution
    ] = Field(
        default_factory=list
    )

    # ========================================================
    # Step 9 bounded revision state
    # ========================================================

    revision_count: int = Field(
        default=0,
        ge=0,
    )

    max_revisions: int = Field(
        default=2,
        ge=0,
    )

    revision_feedback: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    revision_limit_reached: bool = False