from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from langgraph_compiler import compile_workflow

from schemas.state import (
    WorkflowState,
)

from schemas.workflow import (
    WorkflowDefinition,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    __file__
).resolve().parent


WORKFLOW_PATH = (
    ROOT
    /
    "workflows"
    /
    "predictive_modeling.yaml"
)


TASK_PATH = (
    ROOT
    /
    "inputs"
    /
    "task.md"
)


FEATURE_POLICY_PATH = (
    ROOT
    /
    "inputs"
    /
    "feature_policy.yaml"
)


DATASET_PATH = (
    ROOT
    /
    "inputs"
    /
    "data.csv"
)


RUN_DIR = (
    ROOT
    /
    "runs"
    /
    "step9"
)


# ============================================================
# Step 9 configuration
# ============================================================

MAX_REVISIONS = 2


# ============================================================
# Generic file helpers
# ============================================================

def load_yaml(
    path: Path,
) -> Any:
    """
    Load a YAML file.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(
            file
        )


def load_text(
    path: Path,
) -> str:
    """
    Load a UTF-8 text file.
    """

    return path.read_text(
        encoding="utf-8"
    )


def save_json(
    path: Path,
    value: Any,
):
    """
    Save JSON in a human-readable form.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            value,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def save_text(
    path: Path,
    value: str,
):
    """
    Save plain text.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        value,
        encoding="utf-8",
    )


# ============================================================
# Workflow loading
# ============================================================

def load_workflow(
    path: Path,
) -> WorkflowDefinition:
    """
    Load and validate workflow YAML.
    """

    raw = load_yaml(
        path
    )

    return WorkflowDefinition.model_validate(
        raw
    )


# ============================================================
# Feature-policy loading
# ============================================================

def load_feature_policy(
    path: Path,
) -> dict[str, Any]:
    """
    Load feature_policy.yaml and return ONLY the inner
    feature-rule mapping.

    The YAML document has this shape:

        prediction_time: construction_start

        features:
          home_id:
            decision: identifier
            reason: ...

          community:
            decision: allow
            reason: ...

    WorkflowState.feature_policy expects:

        {
            "home_id": {
                "decision": "identifier",
                "reason": "..."
            },
            "community": {
                "decision": "allow",
                "reason": "..."
            }
        }

    Therefore we validate the outer YAML envelope here,
    then return raw["features"].
    """

    raw = load_yaml(
        path
    )


    # --------------------------------------------------------
    # Validate top-level document
    # --------------------------------------------------------

    if not isinstance(
        raw,
        dict,
    ):

        raise ValueError(
            (
                "feature_policy.yaml must contain "
                "a top-level mapping."
            )
        )


    # --------------------------------------------------------
    # Validate prediction-time metadata
    # --------------------------------------------------------

    prediction_time = raw.get(
        "prediction_time"
    )


    if (
        prediction_time
        !=
        "construction_start"
    ):

        raise ValueError(
            (
                "feature_policy.yaml must declare:\n"
                "prediction_time: construction_start\n\n"
                f"Found: {prediction_time!r}"
            )
        )


    # --------------------------------------------------------
    # Extract inner feature mapping
    # --------------------------------------------------------

    features = raw.get(
        "features"
    )


    if not isinstance(
        features,
        dict,
    ):

        raise ValueError(
            (
                "feature_policy.yaml must contain "
                "a top-level 'features' mapping."
            )
        )


    # --------------------------------------------------------
    # Basic validation of every feature rule
    # --------------------------------------------------------

    for (
        feature_name,
        rule,
    ) in features.items():

        if not isinstance(
            rule,
            dict,
        ):

            raise ValueError(
                (
                    f"Feature '{feature_name}' must "
                    "have a mapping containing "
                    "'decision' and 'reason'."
                )
            )


        if (
            "decision"
            not in rule
        ):

            raise ValueError(
                (
                    f"Feature '{feature_name}' "
                    "is missing 'decision'."
                )
            )


        if (
            "reason"
            not in rule
        ):

            raise ValueError(
                (
                    f"Feature '{feature_name}' "
                    "is missing 'reason'."
                )
            )


    return features


# ============================================================
# Dataset profiling
# ============================================================

def build_dataset_facts(
    dataset_path: Path,
) -> dict[str, Any]:
    """
    Build authoritative deterministic dataset facts using
    the profiler created in previous tutorial steps.
    """

    import dataset_profile


    # --------------------------------------------------------
    # Support the profiler function name used in your project.
    # --------------------------------------------------------

    if hasattr(
        dataset_profile,
        "profile_dataset",
    ):

        facts = (
            dataset_profile
            .profile_dataset(
                dataset_path
            )
        )


    elif hasattr(
        dataset_profile,
        "build_dataset_profile",
    ):

        facts = (
            dataset_profile
            .build_dataset_profile(
                dataset_path
            )
        )


    elif hasattr(
        dataset_profile,
        "build_dataset_facts",
    ):

        facts = (
            dataset_profile
            .build_dataset_facts(
                dataset_path
            )
        )


    else:

        raise AttributeError(
            (
                "Could not find a supported "
                "dataset profiling function inside "
                "dataset_profile.py.\n\n"
                "Expected one of:\n"
                "- profile_dataset\n"
                "- build_dataset_profile\n"
                "- build_dataset_facts"
            )
        )


    # --------------------------------------------------------
    # Normalize Pydantic result if necessary
    # --------------------------------------------------------

    if hasattr(
        facts,
        "model_dump",
    ):

        facts = facts.model_dump()


    if not isinstance(
        facts,
        dict,
    ):

        raise TypeError(
            (
                "Dataset profiler must return "
                "a dictionary or Pydantic model."
            )
        )


    return facts


# ============================================================
# Domain state -> GraphState
# ============================================================

def workflow_state_to_graph_state(
    state: WorkflowState,
) -> dict[str, Any]:
    """
    Convert validated Pydantic domain state into
    the dictionary used by LangGraph.
    """

    return state.model_dump()


# ============================================================
# GraphState -> validated domain state
# ============================================================

def graph_state_to_workflow_state(
    state: dict[str, Any],
) -> WorkflowState:
    """
    Revalidate LangGraph's final state with Pydantic.
    """

    return WorkflowState.model_validate(
        state
    )


# ============================================================
# Mermaid graph helper
# ============================================================

def save_graph_mermaid(
    graph,
    path: Path,
):
    """
    Save a Mermaid representation of the graph.

    Graph visualization failure should not stop execution.
    """

    try:

        mermaid = (
            graph
            .get_graph()
            .draw_mermaid()
        )


        save_text(
            path,
            mermaid,
        )


    except Exception as exc:

        save_text(
            path,
            (
                "Could not generate Mermaid graph.\n\n"
                f"{type(exc).__name__}: {exc}\n"
            ),
        )


# ============================================================
# Save agent artifacts
# ============================================================

def save_agent_artifacts(
    final_state: WorkflowState,
):
    """
    Save the latest artifact produced by every agent.

    In Step 9, repeated passes overwrite the agent artifact
    in shared state. Therefore these JSON files represent
    the latest version reached by the workflow.
    """

    for (
        artifact_name,
        artifact,
    ) in final_state.artifacts.items():


        # ----------------------------------------------------
        # Structured output
        # ----------------------------------------------------

        if (
            artifact.structured_data
            is not None
        ):

            value = (
                artifact.structured_data
            )


        # ----------------------------------------------------
        # Unstructured fallback
        # ----------------------------------------------------

        else:

            value = {

                "agent":
                    artifact.agent,

                "skill_version":
                    artifact.skill_version,

                "content":
                    artifact.content,

                "schema_name":
                    artifact.schema_name,
            }


        save_json(
            RUN_DIR
            /
            f"{artifact_name}.json",
            value,
        )


# ============================================================
# Revision summary
# ============================================================

def build_revision_summary(
    final_state: WorkflowState,
) -> dict[str, Any]:
    """
    Produce a small summary describing how the bounded
    Step-9 revision process ended.
    """

    critic_artifact = (
        final_state
        .artifacts
        .get(
            "critic"
        )
    )


    critic_data: dict[
        str,
        Any,
    ] = {}


    if (
        critic_artifact
        is not None
        and
        critic_artifact.structured_data
        is not None
    ):

        critic_data = (
            critic_artifact
            .structured_data
        )


    return {

        "revision_count":
            final_state.revision_count,

        "max_revisions":
            final_state.max_revisions,

        "revision_limit_reached":
            final_state.revision_limit_reached,

        "final_critic_verdict":
            critic_data.get(
                "verdict"
            ),

        "final_recommended_next_action":
            critic_data.get(
                "recommended_next_action"
            ),

        "final_issue_count":
            len(
                critic_data.get(
                    "issues",
                    [],
                )
            ),
    }


# ============================================================
# Console summary
# ============================================================

def print_final_summary(
    final_state: WorkflowState,
):
    """
    Print final Step-9 execution status.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "STEP 9 COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        "Revision count:",
        final_state.revision_count,
    )


    print(
        "Maximum revisions:",
        final_state.max_revisions,
    )


    print(
        "Revision limit reached:",
        final_state.revision_limit_reached,
    )


    print(
        "Artifacts:",
        sorted(
            final_state.artifacts.keys()
        ),
    )


    print(
        "Tool executions:",
        len(
            final_state.tool_log
        ),
    )


    critic_artifact = (
        final_state
        .artifacts
        .get(
            "critic"
        )
    )


    if (
        critic_artifact
        is not None
        and
        critic_artifact.structured_data
        is not None
    ):

        critic_data = (
            critic_artifact
            .structured_data
        )


        print(
            "Final Critic verdict:",
            critic_data.get(
                "verdict"
            ),
        )


        print(
            "Final Critic action:",
            critic_data.get(
                "recommended_next_action"
            ),
        )


        print(
            "Final Critic issues:",
            len(
                critic_data.get(
                    "issues",
                    [],
                )
            ),
        )


    print()

    print(
        "Run directory:"
    )

    print(
        RUN_DIR
    )


# ============================================================
# Main
# ============================================================

def main():
    """
    STEP 9

    Bounded conditional revision workflow.

        Planner
            |
            v
        Statistician
            |
            v
        Critic
            |
            +---- accept -------------> END
            |
            +---- revise_planner -----> Planner
            |
            +---- revise_statistician -> Statistician
            |
            +---- revise_both --------> Planner

    Revision loops are limited by MAX_REVISIONS.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "STEP 9 — BOUNDED CONDITIONAL REVISION LOOP"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # Prepare output directory
    # ========================================================

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # 1. Load workflow
    # ========================================================

    print()
    print(
        "Loading workflow..."
    )


    workflow = load_workflow(
        WORKFLOW_PATH
    )


    print(
        "Workflow:",
        workflow.name,
    )


    print(
        "Version:",
        workflow.version,
    )


    print(
        "Nodes:",
        list(
            workflow.nodes.keys()
        ),
    )


    print(
        "Start node:",
        workflow.start,
    )


    print(
        "Configured end node:",
        workflow.end,
    )


    # ========================================================
    # 2. Load task
    # ========================================================

    print()
    print(
        "Loading task..."
    )


    task = load_text(
        TASK_PATH
    )


    # ========================================================
    # 3. Profile dataset
    # ========================================================

    print(
        "Profiling dataset..."
    )


    dataset_facts = (
        build_dataset_facts(
            DATASET_PATH
        )
    )


    print(
        "Rows:",
        dataset_facts.get(
            "row_count"
        ),
    )


    print(
        "Columns:",
        dataset_facts.get(
            "column_count"
        ),
    )


    # ========================================================
    # 4. Load and normalize feature policy
    # ========================================================

    print(
        "Loading feature policy..."
    )


    feature_policy = (
        load_feature_policy(
            FEATURE_POLICY_PATH
        )
    )


    print(
        "Feature rules loaded:",
        len(
            feature_policy
        ),
    )


    # ========================================================
    # 5. Construct validated initial WorkflowState
    # ========================================================

    print(
        "Building initial state..."
    )


    initial_domain_state = WorkflowState(

        task=task,

        dataset_path=str(
            DATASET_PATH
        ),

        dataset_facts=
            dataset_facts,

        # IMPORTANT:
        #
        # This is now ONLY raw["features"],
        # not the whole YAML document.
        feature_policy=
            feature_policy,

        artifacts={},

        warnings=[],

        tool_log=[],

        # ----------------------------------------------------
        # Step-9 revision control
        # ----------------------------------------------------

        revision_count=0,

        max_revisions=
            MAX_REVISIONS,

        revision_feedback={},

        revision_limit_reached=False,
    )


    print(
        "Initial WorkflowState validated."
    )


    # ========================================================
    # 6. Convert Pydantic state to GraphState
    # ========================================================

    initial_state = (
        workflow_state_to_graph_state(
            initial_domain_state
        )
    )


    # ========================================================
    # 7. Compile LangGraph
    # ========================================================

    print()
    print(
        "Compiling LangGraph..."
    )


    graph = compile_workflow(
        workflow
    )


    print(
        "LangGraph compiled."
    )


    # ========================================================
    # 8. Save workflow snapshots
    # ========================================================

    workflow_snapshot = (
        workflow.model_dump(
            by_alias=True
        )
    )


    save_json(
        RUN_DIR
        /
        "workflow.json",
        workflow_snapshot,
    )


    save_text(
        RUN_DIR
        /
        "workflow.yaml",
        WORKFLOW_PATH.read_text(
            encoding="utf-8"
        ),
    )


    # ========================================================
    # 9. Save Mermaid graph
    # ========================================================

    save_graph_mermaid(
        graph,
        RUN_DIR
        /
        "graph.mmd",
    )


    # ========================================================
    # 10. Execute LangGraph
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "EXECUTING GRAPH"
    )

    print(
        "=" * 70
    )


    graph_result = (
        graph.invoke(
            initial_state
        )
    )


    # ========================================================
    # 11. Revalidate final state
    # ========================================================

    final_state = (
        graph_state_to_workflow_state(
            graph_result
        )
    )


    # ========================================================
    # 12. Save full final state
    # ========================================================

    save_json(
        RUN_DIR
        /
        "state.json",
        final_state.model_dump(),
    )


    # ========================================================
    # 13. Save tool log
    # ========================================================

    save_json(
        RUN_DIR
        /
        "tool_log.json",
        [
            item.model_dump()
            for item
            in final_state.tool_log
        ],
    )


    # ========================================================
    # 14. Save latest agent artifacts
    # ========================================================

    save_agent_artifacts(
        final_state
    )


    # ========================================================
    # 15. Save latest Critic feedback
    # ========================================================

    save_json(
        RUN_DIR
        /
        "revision_feedback.json",
        final_state.revision_feedback,
    )


    # ========================================================
    # 16. Save revision summary
    # ========================================================

    revision_summary = (
        build_revision_summary(
            final_state
        )
    )


    save_json(
        RUN_DIR
        /
        "revision_summary.json",
        revision_summary,
    )


    # ========================================================
    # 17. Console summary
    # ========================================================

    print_final_summary(
        final_state
    )


# ============================================================
# Entry point
# ============================================================

if (
    __name__
    ==
    "__main__"
):

    main()