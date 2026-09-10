from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from checkpointing import (
    create_checkpoint_resources,
)

from checkpoint_history import (
    get_checkpoint_id,
    require_exact_checkpoint,
    save_json,
    snapshot_to_full_record,
)

from run_step11 import (
    CHECKPOINT_DB_PATH,
    build_persistent_graph,
    build_thread_config,
    checkpoint_exists,
    get_thread_run_dir,
    load_workflow,
)

from schemas.state import (
    WorkflowState,
)


# ============================================================
# STEP 11D-C
#
# CONTROLLED COUNTERFACTUAL STATE EDITING
#
# Example:
#
# python checkpoint_counterfactual.py branch \
#     construction-demo-001 \
#     "REAL_PLANNER_CHECKPOINT_ID" \
#     --max-revisions 0
#
# ============================================================


# ============================================================
# Exact config
# ============================================================

def build_exact_config(
    *,
    thread_id: str,
    checkpoint_id: str,
    checkpoint_ns: str = "",
) -> dict[str, Any]:
    """
    Select one exact checkpoint.
    """

    clean_thread_id = (
        str(
            thread_id
        )
        .strip()
    )

    clean_checkpoint_id = (
        str(
            checkpoint_id
        )
        .strip()
    )


    if not clean_thread_id:

        raise ValueError(
            "thread_id cannot be empty."
        )


    if not clean_checkpoint_id:

        raise ValueError(
            "checkpoint_id cannot be empty."
        )


    return {

        "configurable": {

            "thread_id":
                clean_thread_id,

            "checkpoint_ns":
                checkpoint_ns,

            "checkpoint_id":
                clean_checkpoint_id,
        }
    }


# ============================================================
# Exact-checkpoint existence
# ============================================================

def exact_checkpoint_exists(
    resources,
    config: dict[str, Any],
) -> bool:
    """
    True only when this exact checkpoint exists.
    """

    return (
        resources
        .checkpointer
        .get_tuple(
            config
        )
        is not None
    )


# ============================================================
# Snapshot summary
# ============================================================

def summarize_snapshot(
    snapshot,
) -> dict[str, Any]:
    """
    Produce a compact description of graph state.
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


    artifacts = (
        values.get(
            "artifacts",
            {},
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


    return {

        "checkpoint_id":
            get_checkpoint_id(
                snapshot
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
    }


# ============================================================
# Console summary
# ============================================================

def print_snapshot_summary(
    title: str,
    summary: dict[str, Any],
) -> None:
    """
    Print checkpoint summary.
    """

    print()

    print(
        "=" * 70
    )

    print(
        title
    )

    print(
        "=" * 70
    )


    print(
        "Checkpoint ID:",
        summary.get(
            "checkpoint_id"
        ),
    )


    print(
        "Created:",
        summary.get(
            "created_at"
        ),
    )


    print(
        "Next nodes:",
        summary.get(
            "next_nodes"
        ),
    )


    print(
        "Artifacts:",
        summary.get(
            "artifact_names"
        ),
    )


    print(
        "Revision count:",
        summary.get(
            "revision_count"
        ),
    )


    print(
        "Max revisions:",
        summary.get(
            "max_revisions"
        ),
    )


    print(
        "Revision limit reached:",
        summary.get(
            "revision_limit_reached"
        ),
    )


# ============================================================
# Validate Planner-era source checkpoint
# ============================================================

def validate_planner_source(
    snapshot,
) -> None:
    """
    Step 11D-C deliberately operates on the Planner-era
    checkpoint.

    Expected state:

        planner artifact exists

        critic artifact does not exist

        next nodes include:
            researcher
            statistician

    This prevents accidentally applying our counterfactual
    edit to an unrelated checkpoint.
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


    artifacts = (
        values.get(
            "artifacts",
            {},
        )
        or
        {}
    )


    next_nodes = set(
        getattr(
            snapshot,
            "next",
            (),
        )
        or
        ()
    )


    if (
        "planner"
        not in artifacts
    ):

        raise RuntimeError(
            (
                "Selected checkpoint is not a "
                "Planner-era checkpoint because "
                "the Planner artifact is absent."
            )
        )


    if (
        "critic"
        in artifacts
    ):

        raise RuntimeError(
            (
                "Selected checkpoint already contains "
                "a Critic artifact. Choose the earlier "
                "Planner checkpoint."
            )
        )


    expected_next = {
        "researcher",
        "statistician",
    }


    if not expected_next.issubset(
        next_nodes
    ):

        raise RuntimeError(
            (
                "Selected checkpoint does not schedule "
                "the expected parallel branches.\n\n"
                "Expected next nodes to include:\n"
                "    researcher\n"
                "    statistician\n\n"
                f"Actual next nodes:\n"
                f"    {sorted(next_nodes)}"
            )
        )


# ============================================================
# Validate patched state with our Pydantic domain model
# ============================================================

def validate_patched_state(
    snapshot,
) -> WorkflowState:
    """
    Never continue execution from an edited checkpoint until
    the full resulting state still satisfies WorkflowState.
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


    return (
        WorkflowState
        .model_validate(
            values
        )
    )


# ============================================================
# Counterfactual branch
# ============================================================

def command_branch(
    *,
    thread_id: str,
    checkpoint_id: str,
    database_path: Path,
    max_revisions: int,
) -> None:
    """
    Create an alternative future from a historical Planner
    checkpoint by editing max_revisions.

    Example:

        original:
            max_revisions = 2

        counterfactual:
            max_revisions = 0

    The original checkpoint chain remains stored.
    """

    resources = (
        create_checkpoint_resources(
            database_path
        )
    )


    try:

        # ====================================================
        # Compile persistent graph
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


        latest_config = (
            build_thread_config(
                thread_id
            )
        )


        # ====================================================
        # Thread must exist
        # ====================================================

        if not checkpoint_exists(
            resources,
            latest_config,
        ):

            raise RuntimeError(
                (
                    f"Thread '{thread_id}' "
                    "does not exist."
                )
            )


        # ====================================================
        # Verify exact historical source
        # ====================================================

        source_tuple = (
            require_exact_checkpoint(
                resources,
                thread_id=
                    thread_id,
                checkpoint_id=
                    checkpoint_id,
            )
        )


        source_config = (
            source_tuple.config
        )


        source_snapshot = (
            graph.get_state(
                source_config
            )
        )


        validate_planner_source(
            source_snapshot
        )


        source_summary = (
            summarize_snapshot(
                source_snapshot
            )
        )


        print_snapshot_summary(
            "SOURCE PLANNER CHECKPOINT",
            source_summary,
        )


        # ====================================================
        # Capture original current head
        # ====================================================

        original_head_snapshot = (
            graph.get_state(
                latest_config
            )
        )


        original_head_summary = (
            summarize_snapshot(
                original_head_snapshot
            )
        )


        original_head_checkpoint_id = (
            original_head_summary[
                "checkpoint_id"
            ]
        )


        print_snapshot_summary(
            "ORIGINAL THREAD HEAD",
            original_head_summary,
        )


        # ====================================================
        # Original source value
        # ====================================================

        source_values = (
            getattr(
                source_snapshot,
                "values",
                {},
            )
            or
            {}
        )


        original_max_revisions = int(
            source_values.get(
                "max_revisions",
                0,
            )
        )


        print()

        print(
            "=" * 70
        )

        print(
            "COUNTERFACTUAL POLICY"
        )

        print(
            "=" * 70
        )


        print(
            "Original max_revisions:",
            original_max_revisions,
        )


        print(
            "Counterfactual max_revisions:",
            max_revisions,
        )


        # ====================================================
        # Require an actual intervention
        # ====================================================

        if (
            max_revisions
            ==
            original_max_revisions
        ):

            raise RuntimeError(
                (
                    "Counterfactual value equals the "
                    "historical value. Nothing would "
                    "actually be changed."
                )
            )


        # ====================================================
        # CONTROLLED STATE UPDATE
        #
        # IMPORTANT:
        #
        # We update ONLY LastValue channels.
        #
        # We do NOT touch:
        #
        #   artifacts
        #   warnings
        #   tool_log
        #
        # because those have reducer semantics.
        #
        # as_node='planner' means:
        #
        #   behave as though Planner has just completed,
        #   therefore schedule Planner's downstream edges.
        #
        # update_state returns the configuration for the NEW
        # checkpoint it creates.
        # ====================================================

        patch_values = {

            "max_revisions":
                max_revisions,

            # Since this branch begins immediately after
            # Planner, graph-level revision state should be
            # clean.
            "revision_count":
                0,

            "revision_limit_reached":
                False,

            "revision_feedback":
                {},
        }


        print()

        print(
            "Creating counterfactual checkpoint..."
        )


        patched_config = (
            graph.update_state(
                source_config,
                patch_values,
                as_node="planner",
            )
        )


        # ====================================================
        # Inspect new PATCH checkpoint
        # ====================================================

        patched_snapshot = (
            graph.get_state(
                patched_config
            )
        )


        patched_summary = (
            summarize_snapshot(
                patched_snapshot
            )
        )


        patched_checkpoint_id = (
            patched_summary[
                "checkpoint_id"
            ]
        )


        print_snapshot_summary(
            "COUNTERFACTUAL PATCH CHECKPOINT",
            patched_summary,
        )


        # ====================================================
        # Patch must create a new checkpoint
        # ====================================================

        if (
            patched_checkpoint_id
            ==
            checkpoint_id
        ):

            raise RuntimeError(
                (
                    "update_state() did not produce "
                    "a new checkpoint."
                )
            )


        # ====================================================
        # Verify patch actually took effect
        # ====================================================

        patched_values = (
            getattr(
                patched_snapshot,
                "values",
                {},
            )
            or
            {}
        )


        observed_max_revisions = (
            patched_values.get(
                "max_revisions"
            )
        )


        if (
            observed_max_revisions
            !=
            max_revisions
        ):

            raise RuntimeError(
                (
                    "Counterfactual state patch failed.\n\n"
                    f"Requested max_revisions: "
                    f"{max_revisions}\n"
                    f"Observed max_revisions: "
                    f"{observed_max_revisions}"
                )
            )


        # ====================================================
        # Full Pydantic validation BEFORE execution
        # ====================================================

        validated_patch = (
            validate_patched_state(
                patched_snapshot
            )
        )


        print()

        print(
            "Patched WorkflowState validated."
        )


        print(
            "Validated max_revisions:",
            validated_patch.max_revisions,
        )


        # ====================================================
        # Scheduling verification
        # ====================================================

        patched_next = set(
            patched_summary.get(
                "next_nodes",
                [],
            )
        )


        expected_next = {
            "researcher",
            "statistician",
        }


        if not expected_next.issubset(
            patched_next
        ):

            raise RuntimeError(
                (
                    "Counterfactual checkpoint does not "
                    "preserve Planner downstream routing.\n\n"
                    f"Observed next nodes: "
                    f"{sorted(patched_next)}"
                )
            )


        # ====================================================
        # Verify original historical checkpoints are still
        # present BEFORE running alternative future.
        # ====================================================

        source_preserved_before = (
            exact_checkpoint_exists(
                resources,
                source_config,
            )
        )


        original_head_config = (
            build_exact_config(
                thread_id=
                    thread_id,

                checkpoint_id=
                    original_head_checkpoint_id,
            )
        )


        original_head_preserved_before = (
            exact_checkpoint_exists(
                resources,
                original_head_config,
            )
        )


        if not source_preserved_before:

            raise RuntimeError(
                "Historical source checkpoint disappeared."
            )


        if not original_head_preserved_before:

            raise RuntimeError(
                "Original final checkpoint disappeared."
            )


        # ====================================================
        # RUN ALTERNATIVE FUTURE
        #
        # None = no new external initial state.
        #
        # patched_config = begin from our edited checkpoint.
        # ====================================================

        print()

        print(
            "=" * 70
        )

        print(
            "EXECUTING COUNTERFACTUAL FUTURE"
        )

        print(
            "=" * 70
        )


        print(
            "Counterfactual checkpoint:",
            patched_checkpoint_id,
        )


        print(
            "Starting nodes:",
            patched_summary[
                "next_nodes"
            ],
        )


        graph.invoke(
            None,
            config=patched_config,
            durability="sync",
        )


        # ====================================================
        # New latest thread head
        # ====================================================

        new_head_snapshot = (
            graph.get_state(
                latest_config
            )
        )


        new_head_summary = (
            summarize_snapshot(
                new_head_snapshot
            )
        )


        new_head_checkpoint_id = (
            new_head_summary[
                "checkpoint_id"
            ]
        )


        print_snapshot_summary(
            "COUNTERFACTUAL FINAL HEAD",
            new_head_summary,
        )


        # ====================================================
        # Verify original history remains addressable
        # ====================================================

        source_preserved_after = (
            exact_checkpoint_exists(
                resources,
                source_config,
            )
        )


        original_head_preserved_after = (
            exact_checkpoint_exists(
                resources,
                original_head_config,
            )
        )


        patch_preserved = (
            exact_checkpoint_exists(
                resources,
                patched_config,
            )
        )


        new_head_created = (
            new_head_checkpoint_id
            !=
            original_head_checkpoint_id
        )


        # ====================================================
        # Save complete snapshots
        # ====================================================

        run_dir = (
            get_thread_run_dir(
                thread_id
            )
        )


        experiment_dir = (
            run_dir
            /
            "counterfactuals"
        )


        experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        source_short = str(
            checkpoint_id
        )[
            :12
        ]


        patch_short = str(
            patched_checkpoint_id
        )[
            :12
        ]


        new_short = str(
            new_head_checkpoint_id
        )[
            :12
        ]


        experiment_name = (
            f"maxrev_"
            f"{original_max_revisions}"
            f"_to_"
            f"{max_revisions}_"
            f"{source_short}"
        )


        report_path = (
            experiment_dir
            /
            (
                experiment_name
                +
                "_report.json"
            )
        )


        source_path = (
            experiment_dir
            /
            (
                experiment_name
                +
                "_source.json"
            )
        )


        patch_path = (
            experiment_dir
            /
            (
                experiment_name
                +
                "_patch.json"
            )
        )


        final_path = (
            experiment_dir
            /
            (
                experiment_name
                +
                "_final.json"
            )
        )


        save_json(
            source_path,
            snapshot_to_full_record(
                source_snapshot
            ),
        )


        save_json(
            patch_path,
            snapshot_to_full_record(
                patched_snapshot
            ),
        )


        save_json(
            final_path,
            snapshot_to_full_record(
                new_head_snapshot
            ),
        )


        # ====================================================
        # Compare original vs counterfactual outputs
        # ====================================================

        original_head_values = (
            getattr(
                original_head_snapshot,
                "values",
                {},
            )
            or
            {}
        )


        new_head_values = (
            getattr(
                new_head_snapshot,
                "values",
                {},
            )
            or
            {}
        )


        original_artifacts = (
            original_head_values.get(
                "artifacts",
                {},
            )
            or
            {}
        )


        counterfactual_artifacts = (
            new_head_values.get(
                "artifacts",
                {},
            )
            or
            {}
        )


        original_critic = (
            original_artifacts
            .get(
                "critic",
                {},
            )
            or
            {}
        )


        counterfactual_critic = (
            counterfactual_artifacts
            .get(
                "critic",
                {},
            )
            or
            {}
        )


        original_critic_data = (
            original_critic.get(
                "structured_data",
                {},
            )
            or
            {}
        )


        counterfactual_critic_data = (
            counterfactual_critic.get(
                "structured_data",
                {},
            )
            or
            {}
        )


        report = {

            "experiment":
                "max_revisions_counterfactual",

            "thread_id":
                thread_id,

            "database_path":
                str(
                    resources.db_path
                ),

            "intervention": {

                "field":
                    "max_revisions",

                "original_value":
                    original_max_revisions,

                "counterfactual_value":
                    max_revisions,

                "as_node":
                    "planner",
            },

            "source_checkpoint":
                source_summary,

            "original_head":
                original_head_summary,

            "patch_checkpoint":
                patched_summary,

            "counterfactual_head":
                new_head_summary,

            "preservation_checks": {

                "source_preserved_before":
                    source_preserved_before,

                "source_preserved_after":
                    source_preserved_after,

                "original_head_preserved_before":
                    original_head_preserved_before,

                "original_head_preserved_after":
                    original_head_preserved_after,

                "patch_checkpoint_preserved":
                    patch_preserved,

                "new_head_created":
                    new_head_created,
            },

            "original_outcome": {

                "revision_count":
                    original_head_values.get(
                        "revision_count"
                    ),

                "critic_verdict":
                    original_critic_data.get(
                        "verdict"
                    ),

                "critic_action":
                    original_critic_data.get(
                        "recommended_next_action"
                    ),

                "critic_issue_count":
                    len(
                        original_critic_data.get(
                            "issues",
                            [],
                        )
                        or
                        []
                    ),
            },

            "counterfactual_outcome": {

                "revision_count":
                    new_head_values.get(
                        "revision_count"
                    ),

                "max_revisions":
                    new_head_values.get(
                        "max_revisions"
                    ),

                "critic_verdict":
                    counterfactual_critic_data.get(
                        "verdict"
                    ),

                "critic_action":
                    counterfactual_critic_data.get(
                        "recommended_next_action"
                    ),

                "critic_issue_count":
                    len(
                        counterfactual_critic_data.get(
                            "issues",
                            [],
                        )
                        or
                        []
                    ),
            },

            "interpretation": (
                "The counterfactual branch was created by "
                "editing a historical Planner-era checkpoint "
                "using LangGraph update_state(), validating "
                "the resulting WorkflowState, and continuing "
                "execution from the new checkpoint. Original "
                "historical checkpoints were preserved."
            ),
        }


        save_json(
            report_path,
            report,
        )


        # ====================================================
        # Final console verification
        # ====================================================

        print()

        print(
            "=" * 70
        )

        print(
            "COUNTERFACTUAL BRANCH VERIFICATION"
        )

        print(
            "=" * 70
        )


        print(
            "Source preserved:",
            source_preserved_after,
        )


        print(
            "Original head preserved:",
            original_head_preserved_after,
        )


        print(
            "Patch checkpoint preserved:",
            patch_preserved,
        )


        print(
            "New head created:",
            new_head_created,
        )


        print()

        print(
            "Original max_revisions:",
            original_max_revisions,
        )


        print(
            "Counterfactual max_revisions:",
            new_head_values.get(
                "max_revisions"
            ),
        )


        print()

        print(
            "Original Critic verdict:",
            original_critic_data.get(
                "verdict"
            ),
        )


        print(
            "Counterfactual Critic verdict:",
            counterfactual_critic_data.get(
                "verdict"
            ),
        )


        print()

        print(
            "Original revision count:",
            original_head_values.get(
                "revision_count"
            ),
        )


        print(
            "Counterfactual revision count:",
            new_head_values.get(
                "revision_count"
            ),
        )


        print()

        print(
            "Experiment report:"
        )


        print(
            report_path
        )


        # ====================================================
        # Hard preservation checks
        # ====================================================

        if not source_preserved_after:

            raise RuntimeError(
                (
                    "Source historical checkpoint was "
                    "not preserved."
                )
            )


        if not original_head_preserved_after:

            raise RuntimeError(
                (
                    "Original completed branch was "
                    "not preserved."
                )
            )


        if not patch_preserved:

            raise RuntimeError(
                (
                    "Counterfactual patch checkpoint "
                    "was not preserved."
                )
            )


        if not new_head_created:

            raise RuntimeError(
                (
                    "Counterfactual execution did not "
                    "create a new final checkpoint."
                )
            )


        print()

        print(
            "STEP 11D-C COUNTERFACTUAL BRANCH PASSED."
        )


    finally:

        resources.close()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build counterfactual CLI.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Step 11D-C controlled historical "
            "state intervention"
        )
    )


    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )


    branch_parser = (
        subparsers.add_parser(
            "branch",
            help=(
                "Create and execute a controlled "
                "counterfactual branch."
            ),
        )
    )


    branch_parser.add_argument(
        "thread_id",
        help=(
            "Persistent LangGraph thread ID."
        ),
    )


    branch_parser.add_argument(
        "checkpoint_id",
        help=(
            "Planner-era historical checkpoint ID."
        ),
    )


    branch_parser.add_argument(
        "--max-revisions",
        type=int,
        required=True,
        help=(
            "Counterfactual maximum graph-level "
            "Critic revision rounds."
        ),
    )


    branch_parser.add_argument(
        "--db",
        default=str(
            CHECKPOINT_DB_PATH
        ),
        help=(
            "SQLite checkpoint database."
        ),
    )


    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    CLI dispatcher.
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


    if (
        args.command
        ==
        "branch"
    ):

        if (
            args.max_revisions
            <
            0
        ):

            parser.error(
                (
                    "--max-revisions must be "
                    "zero or greater."
                )
            )


        command_branch(

            thread_id=
                args.thread_id,

            checkpoint_id=
                args.checkpoint_id,

            database_path=
                database_path,

            max_revisions=
                args.max_revisions,
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
