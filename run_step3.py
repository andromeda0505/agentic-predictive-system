from pathlib import Path
import json

import yaml
from ollama import chat

from dataset_profile import (
    build_dataset_profile,
)

from schemas.state import (
    WorkflowState,
    AgentArtifact,
    FeatureRule,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).parent

INPUTS_DIR = ROOT / "inputs"
SKILLS_DIR = ROOT / "skills"
RUNS_DIR = ROOT / "runs" / "step3"


DATA_PATH = (
    INPUTS_DIR / "data.csv"
)

TASK_PATH = (
    INPUTS_DIR / "task.md"
)

FEATURE_POLICY_PATH = (
    INPUTS_DIR
    / "feature_policy.yaml"
)

PLANNER_SKILL_PATH = (
    SKILLS_DIR / "planner.md"
)

STATISTICIAN_SKILL_PATH = (
    SKILLS_DIR / "statistician.md"
)

STATE_PATH = (
    RUNS_DIR / "state.json"
)


# ============================================================
# Skill loading
# ============================================================

def load_skill(path: Path):

    text = path.read_text(
        encoding="utf-8"
    )

    if not text.startswith("---"):

        raise ValueError(
            f"No YAML front matter: {path}"
        )

    parts = text.split(
        "---",
        2,
    )

    metadata = yaml.safe_load(
        parts[1]
    )

    instructions = (
        parts[2].strip()
    )

    return (
        metadata,
        instructions,
    )

def load_feature_policy(
    path: Path,
) -> dict[str, FeatureRule]:

    raw = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    features = raw[
        "features"
    ]

    return {

        feature_name:
            FeatureRule(**rule)

        for feature_name, rule
        in features.items()
    }

def validate_feature_policy(
    feature_policy,
    dataset_facts,
):

    dataset_columns = set(
        dataset_facts[
            "column_names"
        ]
    )

    policy_columns = set(
        feature_policy.keys()
    )

    unknown_policy_columns = (
        policy_columns
        -
        dataset_columns
    )

    missing_policy_columns = (
        dataset_columns
        -
        policy_columns
    )

    if unknown_policy_columns:

        raise ValueError(
            "Feature policy contains columns "
            "not present in the dataset: "
            f"{sorted(unknown_policy_columns)}"
        )

    if missing_policy_columns:

        raise ValueError(
            "Dataset contains columns with "
            "no feature policy: "
            f"{sorted(missing_policy_columns)}"
        )
    
# ============================================================
# Model output cleanup
# ============================================================

def clean_output(
    text: str,
) -> str:

    if "</think>" in text:

        text = text.split(
            "</think>",
            1,
        )[1]

    return text.strip()


# ============================================================
# Generic agent runner
# ============================================================

def run_agent(
    skill_path: Path,
    message: str,
):

    metadata, instructions = (
        load_skill(
            skill_path
        )
    )

    print()
    print("=" * 70)
    print(
        f"AGENT: "
        f"{metadata['name'].upper()}"
    )
    print("=" * 70)

    response = chat(

        model=metadata["model"],

        messages=[

            {
                "role": "system",
                "content": instructions,
            },

            {
                "role": "user",
                "content": message,
            },

        ],

        think=False,

        stream=False,

        options={
            "temperature": 0.1,
        },
    )

    output = clean_output(
        response.message.content
    )

    artifact = AgentArtifact(

        agent=metadata["name"],

        skill_version=str(
            metadata["version"]
        ),

        content=output,
    )

    return artifact


# ============================================================
# Save State
# ============================================================

def save_state(
    state: WorkflowState,
):

    STATE_PATH.write_text(

        state.model_dump_json(
            indent=2
        ),

        encoding="utf-8",
    )


# ============================================================
# State → grounding context
# ============================================================

def dataset_grounding_text(
    state: WorkflowState,
):

    return json.dumps(
        state.dataset_facts,
        indent=2,
        default=str,
    )


def feature_policy_text(
    state: WorkflowState,
):

    serializable = {

        name:
            rule.model_dump()

        for name, rule
        in state.feature_policy.items()
    }

    return json.dumps(
        serializable,
        indent=2,
    )


# ============================================================
# Main
# ============================================================

def main():

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("STEP 3 — SHARED STATE")
    print("=" * 70)


    # ========================================================
    # Build state from authoritative inputs
    # ========================================================

    task = TASK_PATH.read_text(
        encoding="utf-8"
    )

    print(
        "\nProfiling dataset "
        "deterministically..."
    )

    dataset_facts = (
        build_dataset_profile(
            DATA_PATH
        )
    )

    feature_policy = (
    load_feature_policy(
        FEATURE_POLICY_PATH
        )
    )


    validate_feature_policy(
        feature_policy,
        dataset_facts,
    )


    state = WorkflowState(

    task=task,

    dataset_path=str(
        DATA_PATH
        ),

    dataset_facts=(
        dataset_facts
        ),

    feature_policy=(
        feature_policy
        ),
    )

    save_state(state)

    print(
        "Initial state saved."
    )


    # ========================================================
    # PLANNER
    # ========================================================

    planner_message = f"""
# ORIGINAL TASK

{state.task}


# AUTHORITATIVE DATASET FACTS

{dataset_grounding_text(state)}


# PREDICTION-TIME FEATURE CONTRACT

{feature_policy_text(state)}


# EVIDENCE RULES

AUTHORITATIVE DATASET FACTS describe what Python
measured in the historical CSV.

PREDICTION-TIME FEATURE CONTRACT describes how each
field may be used operationally.

Column existence does NOT imply prediction-time availability.

Feature decisions mean:

allow:
    may be considered as a predictor

block:
    must never be used as a predictor

verify:
    do not use as a production predictor until confirmed

identifier:
    may be used for identity, grouping, joins, or auditing,
    but not as a predictive feature

reference_time:
    may generate calendar, seasonality, and time-trend
    features known at prediction time

training_only:
    may be used for constructing or auditing historical
    training data, but not as a production predictor


# TARGET RULE

The original task defines the target:

construction_cycle_days.

The target is continuous regression.

It may be derived historically from:

construction_end_date - construction_start_date.

Do not replace or redefine the target.


# ASSIGNMENT

Create the predictive-modeling plan.

Do not fit models.

Do not invent performance thresholds.

Do not invent industry benchmarks.

Do not use blocked, verification-pending, identifier,
or training-only variables as predictors.
""".strip()


    planner_artifact = run_agent(

        PLANNER_SKILL_PATH,

        planner_message,
    )


    state.artifacts[
        "planner"
    ] = planner_artifact

    save_state(state)


    (
        RUNS_DIR
        / "planner.md"
    ).write_text(

        planner_artifact.content,

        encoding="utf-8",
    )


    print(
        "Planner added to state."
    )


    # ========================================================
    # STATISTICIAN
    # ========================================================

    statistician_message = f"""
# ORIGINAL TASK

{state.task}


# AUTHORITATIVE DATASET FACTS

{dataset_grounding_text(state)}


# PLANNER ARTIFACT

{state.artifacts["planner"].content}


# PREDICTION-TIME FEATURE CONTRACT

{feature_policy_text(state)}

# EVIDENCE PRIORITY

Use this priority order:

1. ORIGINAL TASK
2. FEATURE CONTRACT
3. DATASET FACTS
4. UPSTREAM AGENT ARTIFACTS

The Planner artifact is an advisory recommendation,
not factual evidence.

If the Planner contradicts dataset facts, the dataset
facts win.

If neither the task nor dataset establishes something,
mark it REQUIRES VERIFICATION.

The target specification inside AUTHORITATIVE DATASET FACTS is also
authoritative.

A physical target column is not required when the target specification
states that the label can be deterministically derived from historical
outcome columns.

Never replace the target defined in the original task.

Do not invent numerical thresholds.

Do not invent expected model performance.

Do not invent industry benchmarks.

No predictive model has been fitted yet.

The feature policy overrides any feature recommendation
made by the Planner.

If the Planner recommends a blocked feature, reject it.

If the Planner recommends a verification-pending feature,
mark it unresolved rather than using it.

Column existence is not evidence of prediction-time availability.


# ASSIGNMENT

Create the statistical and machine-learning analysis
protocol.
""".strip()


    statistician_artifact = run_agent(

        STATISTICIAN_SKILL_PATH,

        statistician_message,
    )


    state.artifacts[
        "statistician"
    ] = statistician_artifact

    save_state(state)


    (
        RUNS_DIR
        / "statistician.md"
    ).write_text(

        statistician_artifact.content,

        encoding="utf-8",
    )


    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 70)

    print(
        "STEP 3 COMPLETED"
    )

    print("=" * 70)

    print()
    print(
        "Shared state:"
    )

    print(
        STATE_PATH.relative_to(ROOT)
    )

    print()
    print(
        "Planner artifact:"
    )

    print(
        (
            RUNS_DIR
            / "planner.md"
        ).relative_to(ROOT)
    )

    print()
    print(
        "Statistician artifact:"
    )

    print(
        (
            RUNS_DIR
            / "statistician.md"
        ).relative_to(ROOT)
    )


if __name__ == "__main__":

    main()
