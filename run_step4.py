from pathlib import Path
from typing import Any
import json
import shutil
import sys

import yaml
from ollama import chat
from pydantic import BaseModel

from dataset_profile import (
    build_dataset_profile,
)

from schemas.state import (
    WorkflowState,
    AgentArtifact,
    FeatureRule,
)

from schemas.workflow import (
    WorkflowDefinition,
)


# ============================================================
# Root
# ============================================================

ROOT = Path(__file__).parent


# ============================================================
# Load YAML
# ============================================================

def load_yaml(
    path: Path,
) -> dict:

    return yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# Workflow loader
# ============================================================

def load_workflow(
    path: Path,
) -> WorkflowDefinition:

    raw = load_yaml(path)

    workflow = WorkflowDefinition(
        **raw
    )

    validate_workflow(
        workflow
    )

    return workflow


# ============================================================
# Workflow validation
# ============================================================

def validate_workflow(
    workflow: WorkflowDefinition,
):

    node_names = set(
        workflow.nodes.keys()
    )

    if workflow.start not in node_names:

        raise ValueError(
            f"Workflow start node "
            f"'{workflow.start}' "
            f"does not exist."
        )

    if workflow.end not in node_names:

        raise ValueError(
            f"Workflow end node "
            f"'{workflow.end}' "
            f"does not exist."
        )

    for edge in workflow.edges:

        if edge.from_node not in node_names:

            raise ValueError(
                f"Unknown edge source: "
                f"{edge.from_node}"
            )

        if edge.to_node not in node_names:

            raise ValueError(
                f"Unknown edge destination: "
                f"{edge.to_node}"
            )

    # Step 4 deliberately supports only
    # zero or one outgoing edge per node.
    # Branching comes later.

    for node_name in node_names:

        outgoing = [

            edge

            for edge in workflow.edges

            if edge.from_node
            == node_name
        ]

        if len(outgoing) > 1:

            raise ValueError(
                "Step 4 supports only one "
                "outgoing edge per node. "
                f"Node '{node_name}' has "
                f"{len(outgoing)}."
            )


# ============================================================
# Skill loader
# ============================================================

def load_skill(
    path: Path,
):

    text = path.read_text(
        encoding="utf-8"
    )

    if not text.startswith("---"):

        raise ValueError(
            f"Skill lacks YAML front matter: "
            f"{path}"
        )

    parts = text.split(
        "---",
        2,
    )

    if len(parts) != 3:

        raise ValueError(
            f"Invalid skill format: "
            f"{path}"
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


# ============================================================
# Feature policy
# ============================================================

def load_feature_policy(
    path: Path,
) -> dict[str, FeatureRule]:

    raw = load_yaml(
        path
    )

    features = raw[
        "features"
    ]

    return {

        name:
            FeatureRule(**rule)

        for name, rule
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

    unknown = (
        policy_columns
        -
        dataset_columns
    )

    missing = (
        dataset_columns
        -
        policy_columns
    )

    if unknown:

        raise ValueError(
            "Feature policy contains "
            "unknown columns: "
            f"{sorted(unknown)}"
        )

    if missing:

        raise ValueError(
            "Dataset columns missing from "
            "feature policy: "
            f"{sorted(missing)}"
        )


# ============================================================
# State path resolver
# ============================================================

def resolve_state_path(
    state: WorkflowState,
    path: str,
) -> Any:

    parts = path.split(".")

    value: Any = state

    for part in parts:

        if isinstance(
            value,
            BaseModel,
        ):

            if not hasattr(
                value,
                part,
            ):

                raise KeyError(
                    f"State path not found: "
                    f"{path}"
                )

            value = getattr(
                value,
                part,
            )

        elif isinstance(
            value,
            dict,
        ):

            if part not in value:

                raise KeyError(
                    f"State path not found: "
                    f"{path}"
                )

            value = value[
                part
            ]

        else:

            raise KeyError(
                f"Cannot traverse "
                f"'{part}' in path "
                f"'{path}'"
            )

    return value


# ============================================================
# Serialize context safely
# ============================================================

def serialize_value(
    value: Any,
) -> str:

    if isinstance(
        value,
        BaseModel,
    ):

        value = (
            value.model_dump()
        )

    if isinstance(
        value,
        str,
    ):

        return value

    return json.dumps(
        value,
        indent=2,
        default=str,
    )


# ============================================================
# Prompt builder
# ============================================================

def build_agent_message(
    workflow: WorkflowDefinition,
    node_name: str,
    state: WorkflowState,
) -> str:

    node = workflow.nodes[
        node_name
    ]

    sections = []


    # --------------------------------------------------------
    # Grounding rules
    # --------------------------------------------------------

    rules = "\n".join(

        f"- {rule}"

        for rule
        in workflow.grounding_rules
    )

    sections.append(
        "# GLOBAL GROUNDING RULES\n\n"
        + rules
    )


    # --------------------------------------------------------
    # Node-specific context
    # --------------------------------------------------------

    for context_item in node.context:

        value = resolve_state_path(

            state,

            context_item.path,
        )

        sections.append(

            f"# {context_item.title}\n\n"
            +
            serialize_value(
                value
            )
        )


    # --------------------------------------------------------
    # Assignment
    # --------------------------------------------------------

    sections.append(

        "# ASSIGNMENT\n\n"
        +
        node.assignment
    )


    return "\n\n".join(
        sections
    )


# ============================================================
# Clean local model output
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
# Generic agent execution
# ============================================================

def run_node(
    workflow: WorkflowDefinition,
    node_name: str,
    state: WorkflowState,
) -> AgentArtifact:

    node = workflow.nodes[
        node_name
    ]

    skill_path = (
        ROOT
        /
        node.skill
    )

    metadata, instructions = (
        load_skill(
            skill_path
        )
    )

    message = build_agent_message(

        workflow,
        node_name,
        state,
    )

    print()
    print("=" * 70)

    print(
        f"NODE: {node_name}"
    )

    print(
        f"SKILL: {metadata['name']}"
    )

    print(
        f"MODEL: {metadata['model']}"
    )

    print("=" * 70)


    response = chat(

        model=metadata["model"],

        messages=[

            {
                "role": "system",
                "content":
                    instructions,
            },

            {
                "role": "user",
                "content":
                    message,
            },

        ],

        think=False,

        stream=False,

        options={
            "temperature": 0.1,
        },
    )


    result = clean_output(
        response.message.content
    )


    return AgentArtifact(

        agent=metadata["name"],

        skill_version=str(
            metadata["version"]
        ),

        content=result,
    )


# ============================================================
# Edge lookup
# ============================================================

def next_node(
    workflow: WorkflowDefinition,
    current: str,
):

    outgoing = [

        edge

        for edge in workflow.edges

        if edge.from_node
        == current
    ]

    if len(outgoing) == 0:

        return None

    if len(outgoing) > 1:

        raise RuntimeError(
            "Multiple outgoing edges "
            "are not supported in Step 4."
        )

    return outgoing[0].to_node


# ============================================================
# Save state
# ============================================================

def save_state(
    state: WorkflowState,
    path: Path,
):

    path.write_text(

        state.model_dump_json(
            indent=2
        ),

        encoding="utf-8",
    )


# ============================================================
# Main workflow executor
# ============================================================

def main():

    # --------------------------------------------------------
    # Workflow path from command line
    # --------------------------------------------------------

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
    print("=" * 70)
    print("STEP 4 — DECLARATIVE WORKFLOW")
    print("=" * 70)

    print(
        f"\nWorkflow: "
        f"{workflow_path}"
    )


    # --------------------------------------------------------
    # Load workflow
    # --------------------------------------------------------

    workflow = load_workflow(
        workflow_path
    )


    # --------------------------------------------------------
    # Resolve workflow inputs
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Build deterministic state
    # --------------------------------------------------------

    task = task_path.read_text(
        encoding="utf-8"
    )

    print(
        "\nBuilding deterministic "
        "dataset profile..."
    )

    dataset_facts = (
        build_dataset_profile(
            dataset_path
        )
    )

    feature_policy = (
        load_feature_policy(
            feature_policy_path
        )
    )

    validate_feature_policy(

        feature_policy,

        dataset_facts,
    )


    state = WorkflowState(

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
    )


    # --------------------------------------------------------
    # Run directory
    # --------------------------------------------------------

    run_dir = (
        ROOT
        /
        "runs"
        /
        "step4"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    state_path = (
        run_dir
        /
        "state.json"
    )


    save_state(
        state,
        state_path,
    )


    # Save exact workflow used for this run.

    shutil.copy2(

        workflow_path,

        run_dir
        /
        "workflow.yaml",
    )


    # --------------------------------------------------------
    # Execute graph
    # --------------------------------------------------------

    current = (
        workflow.start
    )

    visited = set()


    while current is not None:

        # -----------------------------------------------
        # Guard against accidental cycles for now.
        # Loops are introduced later deliberately.
        # -----------------------------------------------

        if current in visited:

            raise RuntimeError(
                "Cycle detected. "
                "Step 4 supports only "
                "acyclic sequential workflows."
            )

        visited.add(
            current
        )


        # -----------------------------------------------
        # Execute current node
        # -----------------------------------------------

        artifact = run_node(

            workflow,

            current,

            state,
        )


        node_spec = (
            workflow.nodes[
                current
            ]
        )


        # -----------------------------------------------
        # Put output in shared state
        # -----------------------------------------------

        state.artifacts[
            node_spec.output
        ] = artifact


        # -----------------------------------------------
        # Save standalone artifact
        # -----------------------------------------------

        artifact_path = (

            run_dir

            /

            f"{node_spec.output}.md"
        )

        artifact_path.write_text(

            artifact.content,

            encoding="utf-8",
        )


        # -----------------------------------------------
        # Save updated state
        # -----------------------------------------------

        save_state(
            state,
            state_path,
        )


        print(
            f"Saved artifact: "
            f"{artifact_path.relative_to(ROOT)}"
        )


        # -----------------------------------------------
        # Stop at declared end node
        # -----------------------------------------------

        if current == workflow.end:

            break


        # -----------------------------------------------
        # Follow graph edge
        # -----------------------------------------------

        current = next_node(

            workflow,

            current,
        )


        if current is None:

            raise RuntimeError(
                "Workflow terminated before "
                "reaching declared end node."
            )


    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STEP 4 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print("Executed nodes:")

    for node in visited:

        print(
            f"  - {node}"
        )

    print()
    print(
        f"Final state: "
        f"{state_path.relative_to(ROOT)}"
    )

    print(
        f"Workflow snapshot: "
        f"{(run_dir / 'workflow.yaml').relative_to(ROOT)}"
    )


if __name__ == "__main__":

    main()
