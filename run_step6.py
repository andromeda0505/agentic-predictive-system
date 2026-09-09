from pathlib import Path
import shutil
import sys
import json

from dataset_profile import (
    build_dataset_profile,
)

from langgraph_compiler import (
    compile_workflow_to_langgraph,
)

from schemas.state import (
    WorkflowState,
)

from run_step4 import (
    load_workflow,
    load_feature_policy,
    validate_feature_policy,
)


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).parent


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 6 — AGENT TOOLS")
    print("=" * 70)


    # ========================================================
    # Workflow path
    # ========================================================

    if len(sys.argv) > 1:

        workflow_path = Path(
            sys.argv[1]
        )

        if not workflow_path.is_absolute():

            workflow_path = (
                ROOT
                /
                workflow_path
            )

    else:

        workflow_path = (

            ROOT

            /
            "workflows"

            /
            "predictive_modeling.yaml"
        )


    print()
    print(
        f"Workflow: {workflow_path}"
    )


    # ========================================================
    # Load and validate workflow
    # ========================================================

    workflow = load_workflow(
        workflow_path
    )


    # ========================================================
    # Resolve input files
    # ========================================================

    task_path = (

        ROOT

        /
        workflow.inputs[
            "task"
        ]
    )


    dataset_path = (

        ROOT

        /
        workflow.inputs[
            "dataset"
        ]
    )


    feature_policy_path = (

        ROOT

        /
        workflow.inputs[
            "feature_policy"
        ]
    )


    # ========================================================
    # Task
    # ========================================================

    task = task_path.read_text(
        encoding="utf-8"
    )


    # ========================================================
    # Deterministic dataset facts
    # ========================================================

    print()
    print(
        "Building deterministic "
        "dataset profile..."
    )


    dataset_facts = (
        build_dataset_profile(
            dataset_path
        )
    )


    # ========================================================
    # Feature policy
    # ========================================================

    feature_policy = (
        load_feature_policy(
            feature_policy_path
        )
    )


    validate_feature_policy(

        feature_policy,

        dataset_facts,
    )


    # ========================================================
    # Validate initial application state with Pydantic
    # ========================================================

    validated_initial_state = (
        WorkflowState(

            task=task,

            dataset_path=str(
                dataset_path
            ),

            dataset_facts=(
                dataset_facts
            ),

            feature_policy=(
                feature_policy
            ),

            artifacts={},

            warnings=[],

            tool_log=[],
        )
    )


    # Convert Pydantic state into the ordinary dictionaries
    # consumed by our LangGraph TypedDict state.

    initial_state = (
        validated_initial_state
        .model_dump()
    )


    # ========================================================
    # Compile YAML → LangGraph
    # ========================================================

    print()
    print(
        "Compiling workflow "
        "into LangGraph..."
    )


    graph = (
        compile_workflow_to_langgraph(
            workflow
        )
    )


    # ========================================================
    # Run directory
    # ========================================================

    run_dir = (

        ROOT

        /
        "runs"

        /
        "step6"
    )


    run_dir.mkdir(

        parents=True,

        exist_ok=True,
    )


    # ========================================================
    # Save workflow snapshot
    # ========================================================

    shutil.copy2(

        workflow_path,

        run_dir
        /
        "workflow.yaml",
    )


    # ========================================================
    # Save compiled graph visualization
    # ========================================================

    mermaid = (
        graph
        .get_graph()
        .draw_mermaid()
    )


    graph_path = (
        run_dir
        /
        "graph.mmd"
    )


    graph_path.write_text(

        mermaid,

        encoding="utf-8",
    )


    print()
    print(
        "Compiled graph:"
    )

    print()
    print(
        mermaid
    )


    # ========================================================
    # Execute LangGraph
    # ========================================================

    print()
    print("=" * 70)
    print("INVOKING LANGGRAPH")
    print("=" * 70)


    final_state_dict = (
        graph.invoke(
            initial_state
        )
    )


    # ========================================================
    # Validate final state again
    # ========================================================

    final_state = (
        WorkflowState.model_validate(
            final_state_dict
        )
    )

    tool_log_path = (
        run_dir
        /
        "tool_log.json"
    )

    tool_log_path.write_text(

        json.dumps(

            [
                entry.model_dump()

                for entry
                in final_state.tool_log
            ],

            indent=2,
            default=str,
        ),

        encoding="utf-8",
    )
    
    # ========================================================
    # Save state
    # ========================================================

    state_path = (
        run_dir
        /
        "state.json"
    )


    state_path.write_text(

        final_state.model_dump_json(
            indent=2
        ),

        encoding="utf-8",
    )


    # ========================================================
    # Save agent artifacts
    # ========================================================

    for (
        artifact_name,
        artifact,
    ) in final_state.artifacts.items():

        artifact_path = (

            run_dir

            /

            f"{artifact_name}.md"
        )


        artifact_path.write_text(

            artifact.content,

            encoding="utf-8",
        )


    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 70)

    print(
        "STEP 5 COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)


    print()
    print(
        "Final state:"
    )

    print(
        state_path.relative_to(ROOT)
    )


    print()
    print(
        "Compiled graph definition:"
    )

    print(
        graph_path.relative_to(ROOT)
    )


    print()
    print(
        "Artifacts:"
    )

    print()
    print(
        "Tool execution log:"
    )

    print(
        tool_log_path.relative_to(ROOT)
    )


    for artifact_name in (
        final_state.artifacts
    ):

        print(
            f"  - runs/step5/"
            f"{artifact_name}.md"
        )


if __name__ == "__main__":

    main()
