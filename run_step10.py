from __future__ import annotations

import json
from collections import Counter
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
    "step10"
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
    Load YAML.
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
    Load UTF-8 text.
    """

    return path.read_text(
        encoding="utf-8"
    )


def save_json(
    path: Path,
    value: Any,
):
    """
    Save human-readable JSON.
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
    Load feature_policy.yaml.

    YAML structure:

        prediction_time: construction_start

        features:
          community:
            decision: allow
            reason: ...

    WorkflowState expects only the inner feature mapping.
    """

    raw = load_yaml(
        path
    )


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


    # ========================================================
    # Validate prediction time
    # ========================================================

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
                "feature_policy.yaml must declare "
                "'prediction_time: construction_start'. "
                f"Found: {prediction_time!r}"
            )
        )


    # ========================================================
    # Extract feature rules
    # ========================================================

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
                "a 'features' mapping."
            )
        )


    # ========================================================
    # Basic rule validation
    # ========================================================

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
                    "contain a mapping."
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
    Build deterministic authoritative dataset facts using
    the profiler created in previous steps.
    """

    import dataset_profile


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
                "Could not find a supported dataset "
                "profiling function in dataset_profile.py.\n\n"
                "Expected one of:\n"
                "- profile_dataset\n"
                "- build_dataset_profile\n"
                "- build_dataset_facts"
            )
        )


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
                "or Pydantic model."
            )
        )


    return facts


# ============================================================
# State conversion
# ============================================================

def workflow_state_to_graph_state(
    state: WorkflowState,
) -> dict[str, Any]:
    """
    Convert validated Pydantic state into LangGraph state.
    """

    return state.model_dump()


def graph_state_to_workflow_state(
    state: dict[str, Any],
) -> WorkflowState:
    """
    Revalidate final LangGraph state with Pydantic.
    """

    return WorkflowState.model_validate(
        state
    )


# ============================================================
# Mermaid graph
# ============================================================

def save_graph_mermaid(
    graph,
    path: Path,
):
    """
    Save graph visualization as Mermaid text.

    Visualization failure does not stop workflow execution.
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
    Save the latest artifact from each agent.

    Structured agents:
        planner.json
        statistician.json
        critic.json

    Unstructured Researcher:
        researcher.json
        researcher.txt
    """

    for (
        artifact_name,
        artifact,
    ) in final_state.artifacts.items():


        # ====================================================
        # Structured artifact
        # ====================================================

        if (
            artifact.structured_data
            is not None
        ):

            save_json(
                RUN_DIR
                /
                f"{artifact_name}.json",
                artifact.structured_data,
            )


        # ====================================================
        # Unstructured artifact
        # ====================================================

        else:

            metadata = {

                "agent":
                    artifact.agent,

                "skill_version":
                    artifact.skill_version,

                "schema_name":
                    artifact.schema_name,

                "content":
                    artifact.content,
            }


            save_json(
                RUN_DIR
                /
                f"{artifact_name}.json",
                metadata,
            )


            save_text(
                RUN_DIR
                /
                f"{artifact_name}.txt",
                artifact.content,
            )


# ============================================================
# Critic data helper
# ============================================================

def get_critic_data(
    final_state: WorkflowState,
) -> dict[str, Any]:
    """
    Return latest structured Critic output.
    """

    critic_artifact = (
        final_state
        .artifacts
        .get(
            "critic"
        )
    )


    if (
        critic_artifact
        is None
    ):

        return {}


    if (
        critic_artifact.structured_data
        is None
    ):

        return {}


    return (
        critic_artifact
        .structured_data
    )


# ============================================================
# Revision summary
# ============================================================

def build_revision_summary(
    final_state: WorkflowState,
) -> dict[str, Any]:
    """
    Summarize Step-9 bounded revision behavior while running
    inside the Step-10 parallel graph.
    """

    critic_data = (
        get_critic_data(
            final_state
        )
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
# Step 10 parallel summary
# ============================================================

def build_parallel_summary(
    workflow: WorkflowDefinition,
    final_state: WorkflowState,
) -> dict[str, Any]:
    """
    Verify that declared parallel branches produced artifacts
    before the workflow reached its join output.

    This does not measure wall-clock concurrency.

    It verifies graph-level fan-out/fan-in completion.
    """

    groups: list[
        dict[str, Any]
    ] = []


    for group in workflow.parallel_groups:

        source = (
            group.from_node
        )

        join = (
            group.join
        )


        branch_status: list[
            dict[str, Any]
        ] = []


        for branch_node in group.branches:

            branch_spec = (
                workflow.nodes[
                    branch_node
                ]
            )

            output_name = (
                branch_spec.output
            )

            artifact = (
                final_state
                .artifacts
                .get(
                    output_name
                )
            )


            branch_status.append(
                {

                    "node":
                        branch_node,

                    "output_artifact":
                        output_name,

                    "artifact_present":
                        artifact
                        is not None,

                    "structured":
                        (
                            artifact is not None
                            and
                            artifact.structured_data
                            is not None
                        ),
                }
            )


        join_spec = (
            workflow.nodes[
                join
            ]
        )

        join_output = (
            join_spec.output
        )


        join_artifact = (
            final_state
            .artifacts
            .get(
                join_output
            )
        )


        all_branches_completed = all(

            branch[
                "artifact_present"
            ]

            for branch
            in branch_status
        )


        groups.append(
            {

                "source":
                    source,

                "branches":
                    branch_status,

                "join":
                    join,

                "join_output_artifact":
                    join_output,

                "join_artifact_present":
                    join_artifact
                    is not None,

                "all_branches_completed":
                    all_branches_completed,

                "join_completed_after_required_artifacts_exist":
                    (
                        all_branches_completed
                        and
                        join_artifact
                        is not None
                    ),
            }
        )


    # ========================================================
    # Tool execution counts
    # ========================================================

    tool_counts = Counter(
        item.node

        for item
        in final_state.tool_log
    )


    return {

        "parallel_group_count":
            len(
                workflow.parallel_groups
            ),

        "groups":
            groups,

        "final_artifacts":
            sorted(
                final_state.artifacts.keys()
            ),

        "tool_execution_counts_by_node":
            dict(
                tool_counts
            ),
    }


# ============================================================
# Overall Step 10 verification
# ============================================================

def build_step10_verification(
    parallel_summary: dict[str, Any],
) -> dict[str, Any]:
    """
    Produce a simple PASS/FAIL verification for fan-out and
    fan-in behavior.
    """

    groups = parallel_summary.get(
        "groups",
        [],
    )


    if not groups:

        return {

            "passed":
                False,

            "reason":
                (
                    "No parallel groups were declared "
                    "in the workflow."
                ),
        }


    failures: list[
        str
    ] = []


    for (
        index,
        group,
    ) in enumerate(
        groups,
        start=1,
    ):

        if not group.get(
            "all_branches_completed",
            False,
        ):

            failures.append(
                (
                    f"Parallel group {index}: "
                    "one or more branch artifacts "
                    "are missing."
                )
            )


        if not group.get(
            "join_artifact_present",
            False,
        ):

            failures.append(
                (
                    f"Parallel group {index}: "
                    "join artifact is missing."
                )
            )


    return {

        "passed":
            not failures,

        "failures":
            failures,

        "interpretation":
            (
                "PASS means every declared branch produced "
                "its output artifact and the join node also "
                "completed. This verifies LangGraph-level "
                "fan-out/fan-in execution. It does not prove "
                "that Ollama performed simultaneous GPU "
                "inference."
            ),
    }


# ============================================================
# Console parallel-group display
# ============================================================

def print_parallel_definition(
    workflow: WorkflowDefinition,
):
    """
    Display declarative parallel groups before execution.
    """

    print()
    print(
        "Parallel groups:"
    )


    if not workflow.parallel_groups:

        print(
            "  NONE"
        )

        return


    for (
        index,
        group,
    ) in enumerate(
        workflow.parallel_groups,
        start=1,
    ):

        print(
            (
                f"  {index}. "
                f"{group.from_node} "
                f"-> {group.branches} "
                f"-> {group.join}"
            )
        )


# ============================================================
# Final console summary
# ============================================================

def print_final_summary(
    workflow: WorkflowDefinition,
    final_state: WorkflowState,
    parallel_summary: dict[str, Any],
    verification: dict[str, Any],
):
    """
    Print concise Step-10 completion summary.
    """

    critic_data = (
        get_critic_data(
            final_state
        )
    )


    print()
    print(
        "=" * 70
    )

    print(
        "STEP 10 COMPLETE"
    )

    print(
        "=" * 70
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


    print(
        "Revision count:",
        final_state.revision_count,
    )


    print(
        "Revision limit reached:",
        final_state.revision_limit_reached,
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


    print()
    print(
        "PARALLEL EXECUTION VERIFICATION"
    )


    for group in parallel_summary.get(
        "groups",
        [],
    ):

        print()

        print(
            (
                f"Source: "
                f"{group['source']}"
            )
        )


        for branch in group[
            "branches"
        ]:

            print(
                (
                    "  Branch "
                    f"{branch['node']}: "
                    f"artifact_present="
                    f"{branch['artifact_present']}"
                )
            )


        print(
            (
                "  Join "
                f"{group['join']}: "
                "artifact_present="
                f"{group['join_artifact_present']}"
            )
        )


    print()

    print(
        "STEP 10 PARALLEL TEST:",
        (
            "PASS"
            if verification[
                "passed"
            ]
            else
            "FAIL"
        ),
    )


    if not verification[
        "passed"
    ]:

        for failure in verification.get(
            "failures",
            [],
        ):

            print(
                "  -",
                failure,
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
    STEP 10

    Real LangGraph parallel execution.

                         Researcher
                        /          \\
        Planner --------            -------- Critic
                        \\          /
                         Statistician

    Researcher and Statistician are independent downstream
    branches.

    Critic is the fan-in join and waits for both.

    Step-9 bounded conditional revision routing remains active.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "STEP 10 — PARALLEL FAN-OUT / FAN-IN"
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


    print_parallel_definition(
        workflow
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
    # 4. Load prediction-time feature policy
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
    # 5. Construct initial validated domain state
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
    # 6. Convert domain state to GraphState
    # ========================================================

    initial_state = (
        workflow_state_to_graph_state(
            initial_domain_state
        )
    )


    # ========================================================
    # 7. Compile graph
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
    # 8. Save workflow snapshot
    # ========================================================

    save_json(
        RUN_DIR
        /
        "workflow.json",
        workflow.model_dump(
            by_alias=True
        ),
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
    # 10. Execute graph ONCE
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "EXECUTING PARALLEL GRAPH"
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
    # 12. Save full state
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
    # 14. Save agent artifacts
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
    # 17. Build parallel summary
    # ========================================================

    parallel_summary = (
        build_parallel_summary(
            workflow,
            final_state,
        )
    )


    save_json(
        RUN_DIR
        /
        "parallel_summary.json",
        parallel_summary,
    )


    # ========================================================
    # 18. Verify Step 10
    # ========================================================

    verification = (
        build_step10_verification(
            parallel_summary
        )
    )


    save_json(
        RUN_DIR
        /
        "step10_verification.json",
        verification,
    )


    # ========================================================
    # 19. Print final summary
    # ========================================================

    print_final_summary(
        workflow,
        final_state,
        parallel_summary,
        verification,
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
