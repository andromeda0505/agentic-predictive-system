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
# Step configuration
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
    Save JSON using a human-readable format.
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
    Load and validate the declarative workflow YAML.
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
    Load prediction-time feature policy.

    We keep the raw dictionary here.

    WorkflowState will perform Pydantic validation later.
    """

    raw = load_yaml(
        path
    )

    if not isinstance(
        raw,
        dict,
    ):

        raise ValueError(
            "feature_policy.yaml must contain a mapping."
        )

    return raw


# ============================================================
# Dataset profiling
# ============================================================

def build_dataset_facts(
    dataset_path: Path,
) -> dict[str, Any]:
    """
    Build authoritative deterministic dataset facts.

    We import the existing profiler used by previous steps
    instead of duplicating profiling logic here.
    """

    # --------------------------------------------------------
    # Your existing project already contains
    # dataset_profile.py from Step 3.
    #
    # Import locally so run_step9.py stays easy to understand.
    # --------------------------------------------------------

    import dataset_profile


    # ========================================================
    # Support the profiler function created in the earlier
    # tutorial.
    #
    # We intentionally check the common names so this Step-9
    # runner remains compatible with the earlier project.
    # ========================================================

    if hasattr(
        dataset_profile,
        "profile_dataset",
    ):

        facts = dataset_profile.profile_dataset(
            dataset_path
        )


    elif hasattr(
        dataset_profile,
        "build_dataset_profile",
    ):

        facts = dataset_profile.build_dataset_profile(
            dataset_path
        )


    elif hasattr(
        dataset_profile,
        "build_dataset_facts",
    ):

        facts = dataset_profile.build_dataset_facts(
            dataset_path
        )


    else:

        raise AttributeError(
            (
                "Could not find the dataset profiling "
                "function inside dataset_profile.py. "
                "Expected one of: "
                "profile_dataset, "
                "build_dataset_profile, "
                "build_dataset_facts."
            )
        )


    # --------------------------------------------------------
    # Normalize Pydantic objects if the profiler returns one.
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
                "Dataset profiler must return a dictionary "
                "or a Pydantic model."
            )
        )


    return facts


# ============================================================
# Graph-state conversion
# ============================================================

def workflow_state_to_graph_state(
    state: WorkflowState,
) -> dict[str, Any]:
    """
    Convert validated WorkflowState into the dictionary
    expected by LangGraph.

    This gives us the two-state architecture:

        Pydantic WorkflowState
                ↓
        LangGraph GraphState
                ↓
        Pydantic WorkflowState
    """

    return state.model_dump()


# ============================================================
# Final-state reconstruction
# ============================================================

def graph_state_to_workflow_state(
    state: dict[str, Any],
) -> WorkflowState:
    """
    Validate the final LangGraph state with Pydantic.
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
    Save LangGraph Mermaid representation when available.

    Failure to render Mermaid should not stop workflow
    execution.
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
# Save artifacts
# ============================================================

def save_agent_artifacts(
    final_state: WorkflowState,
):
    """
    Save each structured agent artifact as its own JSON file.

    Expected Step-9 files include:

        planner.json
        statistician.json
        critic.json
    """

    for (
        artifact_name,
        artifact,
    ) in final_state.artifacts.items():

        # ----------------------------------------------------
        # Prefer structured_data.
        # ----------------------------------------------------

        if (
            artifact.structured_data
            is not None
        ):

            value = (
                artifact.structured_data
            )


        # ----------------------------------------------------
        # Fallback for unstructured nodes.
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
# Revision history summary
# ============================================================

def build_revision_summary(
    final_state: WorkflowState,
) -> dict[str, Any]:
    """
    Build a compact machine-readable Step-9 summary.
    """

    critic_artifact = (
        final_state
        .artifacts
        .get(
            "critic"
        )
    )


    critic_data = {}


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
                    []
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
    Print a concise summary after graph execution.
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
                    []
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

    Bounded conditional revision loop.

    Execution:

        Planner
           ↓
        Statistician
           ↓
        Critic
           │
           ├── accept
           │      ↓
           │     END
           │
           ├── revise_planner
           │      ↓
           │   Planner
           │
           ├── revise_statistician
           │      ↓
           │   Statistician
           │
           └── revise_both
                  ↓
               Planner

    The graph is bounded by max_revisions.
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
    # Prepare run directory
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
    # 2. Load authoritative task
    # ========================================================

    print()
    print(
        "Loading task..."
    )


    task = load_text(
        TASK_PATH
    )


    # ========================================================
    # 3. Build deterministic dataset facts
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
    # 4. Load feature policy
    # ========================================================

    print(
        "Loading feature policy..."
    )


    feature_policy = (
        load_feature_policy(
            FEATURE_POLICY_PATH
        )
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

        feature_policy=
            feature_policy,

        artifacts={},

        warnings=[],

        tool_log=[],

        # ----------------------------------------------------
        # Step 9 bounded revision configuration
        # ----------------------------------------------------

        revision_count=0,

        max_revisions=
            MAX_REVISIONS,

        revision_feedback={},

        revision_limit_reached=False,
    )


    # ========================================================
    # 6. Convert validated domain state to graph state
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


    # ========================================================
    # 8. Save workflow snapshot
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


    # Also preserve the exact YAML source.
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
    # 10. Execute graph
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
    # 11. Validate final graph state
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
    # 15. Save revision feedback
    # ========================================================

    save_json(
        RUN_DIR
        /
        "revision_feedback.json",
        final_state.revision_feedback,
    )


    # ========================================================
    # 16. Save Step-9 summary
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
    # 17. Print final summary
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