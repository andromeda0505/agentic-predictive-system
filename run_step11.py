from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from checkpointing import (
    CheckpointResources,
    create_checkpoint_resources,
)

from langgraph_compiler import (
    compile_workflow,
)

from schemas.state import (
    WorkflowState,
)

from schemas.workflow import (
    WorkflowDefinition,
)

from run_step10 import (
    build_dataset_facts,
)


# ============================================================
# STEP 11C
#
# DURABLE LANGGRAPH THREADS
#
# Commands:
#
#   python run_step11.py start THREAD_ID
#
#   python run_step11.py start THREAD_ID \
#       --pause-after planner
#
#   python run_step11.py status THREAD_ID
#
#   python run_step11.py resume THREAD_ID
#
# ============================================================


# ============================================================
# Project paths
# ============================================================

ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parent
)


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


CHECKPOINT_DB_PATH = (
    ROOT
    /
    "runs"
    /
    "checkpoints"
    /
    "predictive_modeling.sqlite"
)


STEP11_RUN_ROOT = (
    ROOT
    /
    "runs"
    /
    "step11"
)


# ============================================================
# Workflow configuration
# ============================================================

MAX_REVISIONS = 2


# ============================================================
# Generic save helpers
# ============================================================

def save_json(
    path: Path,
    value: Any,
) -> None:
    """
    Save JSON in a human-readable format.
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
) -> None:
    """
    Save UTF-8 text.
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
# Safe filesystem name for thread
# ============================================================

def safe_thread_directory_name(
    thread_id: str,
) -> str:
    """
    Convert an arbitrary LangGraph thread_id into a safe
    directory name.

    The hash prevents accidental collisions.

    Example:

        construction-demo-001

    becomes approximately:

        construction-demo-001-a47c8e19
    """

    normalized = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        thread_id.strip(),
    )


    normalized = (
        normalized.strip(
            "._-"
        )
        or
        "thread"
    )


    normalized = normalized[
        :80
    ]


    digest = (
        hashlib.sha256(
            thread_id.encode(
                "utf-8"
            )
        )
        .hexdigest()[
            :8
        ]
    )


    return (
        f"{normalized}-{digest}"
    )


def get_thread_run_dir(
    thread_id: str,
) -> Path:
    """
    Human-readable output directory associated with one
    persistent LangGraph thread.
    """

    return (
        STEP11_RUN_ROOT
        /
        safe_thread_directory_name(
            thread_id
        )
    )


# ============================================================
# Workflow loader
# ============================================================

def load_workflow() -> WorkflowDefinition:
    """
    Load and validate predictive_modeling.yaml.
    """

    raw = yaml.safe_load(
        WORKFLOW_PATH.read_text(
            encoding="utf-8"
        )
    )


    if not isinstance(
        raw,
        dict,
    ):

        raise ValueError(
            (
                "Workflow YAML must contain "
                "a top-level mapping."
            )
        )


    return (
        WorkflowDefinition
        .model_validate(
            raw
        )
    )


# ============================================================
# FEATURE POLICY LOADER
#
# THIS IS THE IMPORTANT CORRECTION
# ============================================================

def load_feature_policy() -> dict[str, Any]:
    """
    Load the prediction-time feature contract.

    Supported YAML formats
    ----------------------

    FORMAT A — wrapped:

        prediction_time: construction_start

        features:
          home_id:
            decision: identifier
            reason: ...

          community:
            decision: allow
            reason: ...

    FORMAT B — already flat:

        home_id:
          decision: identifier
          reason: ...

        community:
          decision: allow
          reason: ...


    WorkflowState expects:

        dict[str, FeatureRule]

    That means it needs:

        {
            "home_id": {...},
            "community": {...},
            ...
        }

    and NOT:

        {
            "prediction_time": "...",
            "features": {...}
        }

    Therefore this function unwraps raw["features"] whenever
    the wrapped YAML format is present.
    """

    # ========================================================
    # Read YAML
    # ========================================================

    raw = yaml.safe_load(
        FEATURE_POLICY_PATH.read_text(
            encoding="utf-8"
        )
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
    # Wrapped format
    # ========================================================

    if (
        "features"
        in raw
    ):

        features = raw[
            "features"
        ]


        if not isinstance(
            features,
            dict,
        ):

            raise ValueError(
                (
                    "feature_policy.yaml contains a "
                    "'features' key, but its value "
                    "is not a mapping."
                )
            )


        prediction_time = raw.get(
            "prediction_time"
        )


        if (
            prediction_time
            is not None
        ):

            print(
                "Feature-policy prediction time:",
                prediction_time,
            )


        feature_policy = features


    # ========================================================
    # Already-flat format
    # ========================================================

    else:

        feature_policy = raw


    # ========================================================
    # Basic validation
    # ========================================================

    if not feature_policy:

        raise ValueError(
            (
                "Feature policy contains "
                "no feature rules."
            )
        )


    for (
        feature_name,
        rule,
    ) in feature_policy.items():

        if not isinstance(
            rule,
            dict,
        ):

            raise ValueError(
                (
                    f"Feature rule '{feature_name}' "
                    "must be a mapping."
                )
            )


        if (
            "decision"
            not in rule
        ):

            raise ValueError(
                (
                    f"Feature rule '{feature_name}' "
                    "is missing required field "
                    "'decision'."
                )
            )


        if (
            "reason"
            not in rule
        ):

            raise ValueError(
                (
                    f"Feature rule '{feature_name}' "
                    "is missing required field "
                    "'reason'."
                )
            )


    return feature_policy


# ============================================================
# Build NEW initial state
# ============================================================

def build_initial_state() -> dict[str, Any]:
    """
    Build authoritative initial state for a NEW thread.

    IMPORTANT:

    This function is used only for:

        start

    It must NOT be called during:

        resume

    Resume restores its state from SQLite.
    """

    # ========================================================
    # Task
    # ========================================================

    print()

    print(
        "Loading task..."
    )


    task = TASK_PATH.read_text(
        encoding="utf-8"
    )


    # ========================================================
    # Dataset facts
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
    # Feature policy
    # ========================================================

    print(
        "Loading feature policy..."
    )


    feature_policy = (
        load_feature_policy()
    )


    print(
        "Feature rules loaded:",
        len(
            feature_policy
        ),
    )


    # ========================================================
    # Domain-state validation
    # ========================================================

    print(
        "Building initial WorkflowState..."
    )


    domain_state = WorkflowState(

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
    # WorkflowState -> GraphState dictionary
    # ========================================================

    return (
        domain_state
        .model_dump()
    )


# ============================================================
# LangGraph persistent thread config
# ============================================================

def build_thread_config(
    thread_id: str,
) -> dict[str, Any]:
    """
    Build LangGraph configuration identifying one durable
    thread.

    Example:

        {
            "configurable": {
                "thread_id":
                    "construction-demo-001"
            }
        }

    The SAME thread_id must be reused to resume the same
    workflow.
    """

    clean_thread_id = (
        str(
            thread_id
        )
        .strip()
    )


    if not clean_thread_id:

        raise ValueError(
            "thread_id cannot be empty."
        )


    return {

        "configurable": {

            "thread_id":
                clean_thread_id,
        }
    }


# ============================================================
# Check whether persistent thread exists
# ============================================================

def checkpoint_exists(
    resources: CheckpointResources,
    config: dict[str, Any],
) -> bool:
    """
    Check SQLite directly for an existing thread checkpoint.

    This protects us against accidentally starting a second
    workflow using an existing thread_id.
    """

    checkpoint_tuple = (
        resources
        .checkpointer
        .get_tuple(
            config
        )
    )


    return (
        checkpoint_tuple
        is not None
    )


# ============================================================
# Snapshot -> simple dictionary
# ============================================================

def snapshot_to_dict(
    snapshot,
    *,
    thread_id: str,
    database_path: Path,
) -> dict[str, Any]:
    """
    Convert LangGraph StateSnapshot into a compact diagnostic
    representation.
    """

    values = (
        getattr(
            snapshot,
            "values",
            {},
        )
        or
        {}
    )


    snapshot_config = (
        getattr(
            snapshot,
            "config",
            {},
        )
        or
        {}
    )


    configurable = (
        snapshot_config.get(
            "configurable",
            {}
        )
        or
        {}
    )


    next_nodes = list(
        getattr(
            snapshot,
            "next",
            (),
        )
        or
        ()
    )


    artifacts = (
        values.get(
            "artifacts",
            {}
        )
        or
        {}
    )


    if next_nodes:

        status = (
            "pending"
        )

    else:

        status = (
            "complete"
        )


    return {

        "thread_id":
            thread_id,

        "database_path":
            str(
                database_path
            ),

        "status":
            status,

        "checkpoint_id":
            configurable.get(
                "checkpoint_id"
            ),

        "created_at":
            getattr(
                snapshot,
                "created_at",
                None,
            ),

        "next_nodes":
            next_nodes,

        "artifact_names":
            sorted(
                artifacts.keys()
            ),

        "revision_count":
            values.get(
                "revision_count"
            ),

        "max_revisions":
            values.get(
                "max_revisions"
            ),

        "revision_limit_reached":
            values.get(
                "revision_limit_reached"
            ),

        "metadata":
            getattr(
                snapshot,
                "metadata",
                None,
            ),
    }


# ============================================================
# Save latest GraphState
# ============================================================

def save_graph_state(
    values: dict[str, Any],
    run_dir: Path,
) -> None:
    """
    Export the latest durable GraphState into ordinary files.

    IMPORTANT:

    SQLite is the persistence mechanism.

    state.json and artifact files are convenience exports for
    humans, debugging, and auditability.
    """

    if not values:

        return


    # ========================================================
    # Validate checkpoint using domain schema
    # ========================================================

    validated_state = (
        WorkflowState
        .model_validate(
            values
        )
    )


    # ========================================================
    # Full state
    # ========================================================

    save_json(
        run_dir
        /
        "state.json",

        validated_state.model_dump(),
    )


    # ========================================================
    # Tool log
    # ========================================================

    save_json(
        run_dir
        /
        "tool_log.json",

        [
            item.model_dump()

            for item
            in validated_state.tool_log
        ],
    )


    # ========================================================
    # Revision feedback
    # ========================================================

    save_json(
        run_dir
        /
        "revision_feedback.json",

        validated_state.revision_feedback,
    )


    # ========================================================
    # Artifacts
    # ========================================================

    artifact_dir = (
        run_dir
        /
        "artifacts"
    )


    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    for (
        artifact_name,
        artifact,
    ) in validated_state.artifacts.items():

        artifact_data = (
            artifact.model_dump()
        )


        # ----------------------------------------------------
        # Complete artifact envelope
        # ----------------------------------------------------

        save_json(
            artifact_dir
            /
            f"{artifact_name}_artifact.json",

            artifact_data,
        )


        # ----------------------------------------------------
        # Human-readable content
        # ----------------------------------------------------

        save_text(
            artifact_dir
            /
            f"{artifact_name}.txt",

            artifact.content,
        )


        # ----------------------------------------------------
        # Structured payload
        # ----------------------------------------------------

        if (
            artifact.structured_data
            is not None
        ):

            save_json(
                artifact_dir
                /
                f"{artifact_name}.json",

                artifact.structured_data,
            )


# ============================================================
# Save stable metadata for persistent thread
# ============================================================

def save_thread_metadata(
    *,
    run_dir: Path,
    workflow: WorkflowDefinition,
    thread_id: str,
    database_path: Path,
) -> None:
    """
    Save stable metadata describing this persistent workflow.
    """

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    save_json(
        run_dir
        /
        "thread.json",

        {

            "thread_id":
                thread_id,

            "checkpoint_database":
                str(
                    database_path
                ),

            "workflow_name":
                workflow.name,

            "workflow_version":
                workflow.version,

            "workflow_path":
                str(
                    WORKFLOW_PATH
                ),

            "dataset_path":
                str(
                    DATASET_PATH
                ),

            "feature_policy_path":
                str(
                    FEATURE_POLICY_PATH
                ),
        },
    )


    # Preserve exact workflow source.
    save_text(
        run_dir
        /
        "workflow.yaml",

        WORKFLOW_PATH.read_text(
            encoding="utf-8"
        ),
    )


    # Preserve feature-contract source too.
    save_text(
        run_dir
        /
        "feature_policy.yaml",

        FEATURE_POLICY_PATH.read_text(
            encoding="utf-8"
        ),
    )


# ============================================================
# Export latest checkpoint
# ============================================================

def persist_snapshot_exports(
    graph,
    config: dict[str, Any],
    *,
    thread_id: str,
    database_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """
    Read latest durable LangGraph checkpoint and write
    human-readable exports.
    """

    snapshot = (
        graph.get_state(
            config
        )
    )


    status = (
        snapshot_to_dict(
            snapshot,
            thread_id=thread_id,
            database_path=database_path,
        )
    )


    save_json(
        run_dir
        /
        "checkpoint_status.json",

        status,
    )


    values = (
        getattr(
            snapshot,
            "values",
            {},
        )
        or
        {}
    )


    save_graph_state(
        values,
        run_dir,
    )


    return status


# ============================================================
# Console status printer
# ============================================================

def print_status(
    status: dict[str, Any],
) -> None:
    """
    Print a concise durable-thread status report.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "PERSISTENT THREAD STATUS"
    )

    print(
        "=" * 70
    )


    print(
        "Thread ID:",
        status.get(
            "thread_id"
        ),
    )


    print(
        "Status:",
        status.get(
            "status"
        ),
    )


    print(
        "Checkpoint ID:",
        status.get(
            "checkpoint_id"
        ),
    )


    print(
        "Next nodes:",
        status.get(
            "next_nodes"
        ),
    )


    print(
        "Artifacts:",
        status.get(
            "artifact_names"
        ),
    )


    print(
        "Revision count:",
        status.get(
            "revision_count"
        ),
    )


    print(
        "Revision limit reached:",
        status.get(
            "revision_limit_reached"
        ),
    )


# ============================================================
# Compile persistence-aware graph
# ============================================================

def build_persistent_graph(
    workflow: WorkflowDefinition,
    resources: CheckpointResources,
):
    """
    Compile our Step-10 graph using the durable SQLite
    checkpointer created in Step 11A.
    """

    print()

    print(
        "Compiling persistence-aware LangGraph..."
    )


    graph = (
        compile_workflow(

            workflow,

            checkpointer=
                resources.checkpointer,
        )
    )


    if not hasattr(
        graph,
        "invoke",
    ):

        raise TypeError(
            (
                "Compiled graph does not support "
                ".invoke(). "
                f"Received type: "
                f"{type(graph).__name__}"
            )
        )


    if not hasattr(
        graph,
        "get_state",
    ):

        raise TypeError(
            (
                "Compiled persistence-aware graph does "
                "not support .get_state()."
            )
        )


    print(
        "Persistence-aware LangGraph compiled."
    )


    return graph


# ============================================================
# START command
# ============================================================

def command_start(
    *,
    thread_id: str,
    database_path: Path,
    pause_after: str | None,
) -> None:
    """
    Start a NEW durable workflow thread.

    Initial state is supplied exactly once.
    """

    resources = (
        create_checkpoint_resources(
            database_path
        )
    )


    graph = None

    config = None


    try:

        # ====================================================
        # Workflow
        # ====================================================

        workflow = (
            load_workflow()
        )


        graph = (
            build_persistent_graph(
                workflow,
                resources,
            )
        )


        # ====================================================
        # Thread identity
        # ====================================================

        config = (
            build_thread_config(
                thread_id
            )
        )


        run_dir = (
            get_thread_run_dir(
                thread_id
            )
        )


        # ====================================================
        # Do not overwrite existing persistent workflow
        # ====================================================

        if checkpoint_exists(
            resources,
            config,
        ):

            raise RuntimeError(
                (
                    f"Thread '{thread_id}' already exists "
                    "in the checkpoint database.\n\n"
                    "Inspect it with:\n\n"
                    f"    python run_step11.py status "
                    f"\"{thread_id}\"\n\n"
                    "Resume it with:\n\n"
                    f"    python run_step11.py resume "
                    f"\"{thread_id}\"\n\n"
                    "Use a new thread_id if you want a "
                    "completely independent workflow."
                )
            )


        # ====================================================
        # Validate test breakpoint
        # ====================================================

        if (
            pause_after
            is not None
            and
            pause_after
            not in workflow.nodes
        ):

            raise ValueError(
                (
                    f"Unknown --pause-after node "
                    f"'{pause_after}'.\n"
                    "Valid nodes: "
                    f"{sorted(workflow.nodes.keys())}"
                )
            )


        # ====================================================
        # Human-readable metadata
        # ====================================================

        save_thread_metadata(
            run_dir=run_dir,
            workflow=workflow,
            thread_id=thread_id,
            database_path=
                resources.db_path,
        )


        # ====================================================
        # Construct initial state
        #
        # THIS IS WHERE THE FIXED FEATURE POLICY LOADER RUNS.
        # ====================================================

        initial_state = (
            build_initial_state()
        )


        print()

        print(
            "=" * 70
        )

        print(
            "STARTING NEW PERSISTENT THREAD"
        )

        print(
            "=" * 70
        )


        print(
            "Thread ID:",
            thread_id,
        )


        print(
            "Checkpoint database:",
            resources.db_path,
        )


        # ====================================================
        # Normal execution
        # ====================================================

        if (
            pause_after
            is None
        ):

            graph.invoke(
                initial_state,
                config=config,

                # --------------------------------------------
                # Persist every checkpoint synchronously
                # before proceeding.
                # --------------------------------------------
                durability="sync",
            )


        # ====================================================
        # Intentional Step-11C pause
        # ====================================================

        else:

            print()

            print(
                (
                    "Intentional checkpoint pause after: "
                    f"{pause_after}"
                )
            )


            graph.invoke(
                initial_state,
                config=config,

                interrupt_after=[
                    pause_after
                ],

                # --------------------------------------------
                # Critical for persistence demonstration:
                # ensure checkpoint reaches disk before exit.
                # --------------------------------------------
                durability="sync",
            )


        # ====================================================
        # Inspect durable state AFTER execution/pause
        # ====================================================

        status = (
            persist_snapshot_exports(
                graph,
                config,
                thread_id=thread_id,
                database_path=
                    resources.db_path,
                run_dir=run_dir,
            )
        )


        print_status(
            status
        )


        if (
            status[
                "status"
            ]
            ==
            "pending"
        ):

            print()

            print(
                (
                    "Workflow is safely checkpointed "
                    "and can be resumed."
                )
            )


            print()

            print(
                "Resume with:"
            )


            print()

            print(
                (
                    f"python run_step11.py resume "
                    f"\"{thread_id}\""
                )
            )


        else:

            print()

            print(
                (
                    "Workflow completed. "
                    "Final state is durable."
                )
            )


    # ========================================================
    # Preserve checkpoint on failure
    # ========================================================

    except Exception:

        # ----------------------------------------------------
        # Never delete persistence because execution failed.
        # ----------------------------------------------------

        try:

            if (
                graph
                is not None
                and
                config
                is not None
                and
                checkpoint_exists(
                    resources,
                    config,
                )
            ):

                run_dir = (
                    get_thread_run_dir(
                        thread_id
                    )
                )


                status = (
                    persist_snapshot_exports(
                        graph,
                        config,
                        thread_id=thread_id,
                        database_path=
                            resources.db_path,
                        run_dir=run_dir,
                    )
                )


                print_status(
                    status
                )


                print()

                print(
                    (
                        "A durable checkpoint was "
                        "preserved despite the error."
                    )
                )


                print()

                print(
                    (
                        "After fixing the problem, resume "
                        "with:"
                    )
                )


                print()

                print(
                    (
                        f"python run_step11.py resume "
                        f"\"{thread_id}\""
                    )
                )


        except Exception:

            # Preserve the ORIGINAL exception.
            pass


        raise


    finally:

        resources.close()


# ============================================================
# RESUME command
# ============================================================

def command_resume(
    *,
    thread_id: str,
    database_path: Path,
) -> None:
    """
    Resume an existing durable workflow.

    THE IMPORTANT OPERATION IS:

        graph.invoke(
            None,
            config=config,
        )

    None means:

        do NOT inject a fresh initial state;
        continue this existing persistent thread.
    """

    resources = (
        create_checkpoint_resources(
            database_path
        )
    )


    graph = None

    config = None


    try:

        # ====================================================
        # Compile same workflow
        # ====================================================

        workflow = (
            load_workflow()
        )


        graph = (
            build_persistent_graph(
                workflow,
                resources,
            )
        )


        # ====================================================
        # SAME thread identity
        # ====================================================

        config = (
            build_thread_config(
                thread_id
            )
        )


        run_dir = (
            get_thread_run_dir(
                thread_id
            )
        )


        # ====================================================
        # Must already exist
        # ====================================================

        if not checkpoint_exists(
            resources,
            config,
        ):

            raise RuntimeError(
                (
                    f"No persistent checkpoint exists "
                    f"for thread '{thread_id}'.\n\n"
                    "Start it first with:\n\n"
                    f"    python run_step11.py start "
                    f"\"{thread_id}\""
                )
            )


        # ====================================================
        # State BEFORE resume
        # ====================================================

        before_snapshot = (
            graph.get_state(
                config
            )
        )


        before_status = (
            snapshot_to_dict(
                before_snapshot,
                thread_id=thread_id,
                database_path=
                    resources.db_path,
            )
        )


        print_status(
            before_status
        )


        # ====================================================
        # Already complete?
        # ====================================================

        if not before_status[
            "next_nodes"
        ]:

            print()

            print(
                (
                    "This persistent thread is already "
                    "complete. No agents were rerun."
                )
            )


            persist_snapshot_exports(
                graph,
                config,
                thread_id=thread_id,
                database_path=
                    resources.db_path,
                run_dir=run_dir,
            )


            return


        # ====================================================
        # Resume
        # ====================================================

        print()

        print(
            "=" * 70
        )

        print(
            "RESUMING FROM DURABLE CHECKPOINT"
        )

        print(
            "=" * 70
        )


        print(
            "Thread ID:",
            thread_id,
        )


        print(
            "Pending nodes:",
            before_status[
                "next_nodes"
            ],
        )


        # ====================================================
        # CRITICAL STEP 11C LINE
        #
        # We do NOT call build_initial_state().
        #
        # SQLite supplies the existing state.
        # ====================================================

        graph.invoke(
            None,
            config=config,
            durability="sync",
        )


        # ====================================================
        # State AFTER resume
        # ====================================================

        after_status = (
            persist_snapshot_exports(
                graph,
                config,
                thread_id=thread_id,
                database_path=
                    resources.db_path,
                run_dir=run_dir,
            )
        )


        print_status(
            after_status
        )


        if (
            after_status[
                "status"
            ]
            ==
            "complete"
        ):

            print()

            print(
                (
                    "Persistent workflow completed "
                    "successfully."
                )
            )


        else:

            print()

            print(
                (
                    "Workflow remains pending and can "
                    "be resumed again using the same "
                    "thread_id."
                )
            )


    # ========================================================
    # Preserve checkpoint if resume itself fails
    # ========================================================

    except Exception:

        try:

            if (
                graph
                is not None
                and
                config
                is not None
                and
                checkpoint_exists(
                    resources,
                    config,
                )
            ):

                run_dir = (
                    get_thread_run_dir(
                        thread_id
                    )
                )


                status = (
                    persist_snapshot_exports(
                        graph,
                        config,
                        thread_id=thread_id,
                        database_path=
                            resources.db_path,
                        run_dir=run_dir,
                    )
                )


                print_status(
                    status
                )


                print()

                print(
                    (
                        "Checkpoint remains durable. "
                        "Fix the error and issue the same "
                        "resume command again."
                    )
                )


        except Exception:

            pass


        raise


    finally:

        resources.close()


# ============================================================
# STATUS command
# ============================================================

def command_status(
    *,
    thread_id: str,
    database_path: Path,
) -> None:
    """
    Inspect persistent workflow WITHOUT executing any node.
    """

    resources = (
        create_checkpoint_resources(
            database_path
        )
    )


    try:

        workflow = (
            load_workflow()
        )


        graph = (
            build_persistent_graph(
                workflow,
                resources,
            )
        )


        config = (
            build_thread_config(
                thread_id
            )
        )


        # ====================================================
        # No thread
        # ====================================================

        if not checkpoint_exists(
            resources,
            config,
        ):

            print()

            print(
                "=" * 70
            )

            print(
                "PERSISTENT THREAD STATUS"
            )

            print(
                "=" * 70
            )


            print(
                "Thread ID:",
                thread_id,
            )


            print(
                "Checkpoint exists: False"
            )


            print(
                (
                    "No workflow state exists "
                    "for this thread."
                )
            )


            return


        # ====================================================
        # Existing thread
        # ====================================================

        run_dir = (
            get_thread_run_dir(
                thread_id
            )
        )


        status = (
            persist_snapshot_exports(
                graph,
                config,
                thread_id=thread_id,
                database_path=
                    resources.db_path,
                run_dir=run_dir,
            )
        )


        print_status(
            status
        )


    finally:

        resources.close()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Step 11C durable LangGraph runner"
        )
    )


    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )


    # ========================================================
    # START
    # ========================================================

    start_parser = (
        subparsers.add_parser(
            "start",
            help=(
                "Start a brand-new persistent workflow."
            ),
        )
    )


    start_parser.add_argument(
        "thread_id",
        help=(
            "Stable persistent workflow identity."
        ),
    )


    start_parser.add_argument(
        "--db",
        default=str(
            CHECKPOINT_DB_PATH
        ),
        help=(
            "SQLite checkpoint database path."
        ),
    )


    start_parser.add_argument(
        "--pause-after",
        default=None,
        help=(
            "Optional test breakpoint. "
            "Example: --pause-after planner"
        ),
    )


    # ========================================================
    # RESUME
    # ========================================================

    resume_parser = (
        subparsers.add_parser(
            "resume",
            help=(
                "Resume an existing persistent workflow."
            ),
        )
    )


    resume_parser.add_argument(
        "thread_id",
        help=(
            "Existing persistent thread ID."
        ),
    )


    resume_parser.add_argument(
        "--db",
        default=str(
            CHECKPOINT_DB_PATH
        ),
        help=(
            "SQLite checkpoint database path."
        ),
    )


    # ========================================================
    # STATUS
    # ========================================================

    status_parser = (
        subparsers.add_parser(
            "status",
            help=(
                "Inspect a thread without executing it."
            ),
        )
    )


    status_parser.add_argument(
        "thread_id",
        help=(
            "Persistent thread ID to inspect."
        ),
    )


    status_parser.add_argument(
        "--db",
        default=str(
            CHECKPOINT_DB_PATH
        ),
        help=(
            "SQLite checkpoint database path."
        ),
    )


    return parser


# ============================================================
# Main CLI dispatcher
# ============================================================

def main() -> None:
    """
    Dispatch start / resume / status.
    """

    parser = (
        build_parser()
    )


    args = (
        parser.parse_args()
    )


    database_path = (
        Path(
            args.db
        )
        .expanduser()
        .resolve()
    )


    # ========================================================
    # START
    # ========================================================

    if (
        args.command
        ==
        "start"
    ):

        command_start(

            thread_id=
                args.thread_id,

            database_path=
                database_path,

            pause_after=
                args.pause_after,
        )


    # ========================================================
    # RESUME
    # ========================================================

    elif (
        args.command
        ==
        "resume"
    ):

        command_resume(

            thread_id=
                args.thread_id,

            database_path=
                database_path,
        )


    # ========================================================
    # STATUS
    # ========================================================

    elif (
        args.command
        ==
        "status"
    ):

        command_status(

            thread_id=
                args.thread_id,

            database_path=
                database_path,
        )


    else:

        parser.error(
            (
                f"Unknown command: "
                f"{args.command}"
            )
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