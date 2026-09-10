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


# ============================================================
# STEP 11D-B
#
# HISTORICAL CHECKPOINT REPLAY
#
# Example:
#
# python checkpoint_replay.py replay \
#     construction-demo-001 \
#     "REAL_CHECKPOINT_ID"
#
# ============================================================


# ============================================================
# Helper: exact checkpoint existence
# ============================================================

def exact_checkpoint_exists(
    resources,
    config: dict[str, Any],
) -> bool:
    """
    Return True only if the exact thread/checkpoint pair exists.
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
# Helper: checkpoint config
# ============================================================

def build_exact_config(
    *,
    thread_id: str,
    checkpoint_id: str,
    checkpoint_ns: str = "",
) -> dict[str, Any]:
    """
    Select one exact checkpoint in one persistent thread.
    """

    clean_thread_id = str(
        thread_id
    ).strip()

    clean_checkpoint_id = str(
        checkpoint_id
    ).strip()


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
# Helper: summarize snapshot
# ============================================================

def summarize_snapshot(
    snapshot,
) -> dict[str, Any]:
    """
    Compact representation of one graph state.
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
            {},
        )
        or
        {}
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

        "revision_limit_reached":
            values.get(
                "revision_limit_reached"
            ),
    }


# ============================================================
# Helper: print snapshot
# ============================================================

def print_snapshot_summary(
    title: str,
    summary: dict[str, Any],
) -> None:
    """
    Print a compact checkpoint description.
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


# ============================================================
# REPLAY
# ============================================================

def command_replay(
    *,
    thread_id: str,
    checkpoint_id: str,
    database_path: Path,
) -> None:
    """
    Replay graph execution from one historical checkpoint.

    IMPORTANT
    ---------

    This does NOT overwrite the selected checkpoint.

    LangGraph creates new checkpoints descending from the
    selected historical checkpoint.

    Therefore:

        old branch remains in SQLite
        +
        new branch is appended

    This is time-travel execution.
    """

    resources = (
        create_checkpoint_resources(
            database_path
        )
    )


    try:

        # ====================================================
        # Load graph
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
        # Base/latest config
        # ====================================================

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
        # Verify selected historical checkpoint physically
        # exists.
        #
        # This also protects us from placeholder text and
        # mistyped checkpoint IDs.
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


        # ====================================================
        # Read SOURCE historical snapshot
        # ====================================================

        source_snapshot = (
            graph.get_state(
                source_config
            )
        )


        source_summary = (
            summarize_snapshot(
                source_snapshot
            )
        )


        print_snapshot_summary(
            "SOURCE HISTORICAL CHECKPOINT",
            source_summary,
        )


        # ====================================================
        # Safety:
        #
        # A terminal checkpoint has nothing scheduled after it.
        # Replaying it would not demonstrate time travel.
        # ====================================================

        if not source_summary[
            "next_nodes"
        ]:

            raise RuntimeError(
                (
                    "\n"
                    "The selected checkpoint is terminal.\n"
                    "\n"
                    "It has no pending next nodes, so there "
                    "is nothing to replay.\n"
                    "\n"
                    "Choose an earlier checkpoint, ideally "
                    "the Planner-era checkpoint with:\n"
                    "\n"
                    "    Artifacts: ['planner']\n"
                    "\n"
                    "and:\n"
                    "\n"
                    "    Next nodes: "
                    "['researcher', 'statistician']\n"
                )
            )


        # ====================================================
        # Capture ORIGINAL HEAD before branching
        # ====================================================

        previous_head_snapshot = (
            graph.get_state(
                latest_config
            )
        )


        previous_head_summary = (
            summarize_snapshot(
                previous_head_snapshot
            )
        )


        previous_head_checkpoint_id = (
            previous_head_summary.get(
                "checkpoint_id"
            )
        )


        print_snapshot_summary(
            "ORIGINAL THREAD HEAD BEFORE REPLAY",
            previous_head_summary,
        )


        # ====================================================
        # Confirm source and old head are different
        # ====================================================

        if (
            source_summary.get(
                "checkpoint_id"
            )
            ==
            previous_head_checkpoint_id
        ):

            raise RuntimeError(
                (
                    "You selected the current thread head. "
                    "That is not historical time travel.\n"
                    "Choose an older checkpoint from "
                    "checkpoint_history.py."
                )
            )


        # ====================================================
        # TIME TRAVEL
        #
        # This is the central Step-11D-B operation.
        #
        # None:
        #     supply no new initial input.
        #
        # source_config contains checkpoint_id:
        #     resume execution FROM THAT historical state.
        #
        # durability='sync':
        #     ensure new checkpoints are durably committed.
        # ====================================================

        print()

        print(
            "=" * 70
        )

        print(
            "STARTING HISTORICAL REPLAY"
        )

        print(
            "=" * 70
        )


        print(
            "Thread ID:",
            thread_id,
        )


        print(
            "Replay source checkpoint:",
            checkpoint_id,
        )


        print(
            "Initially scheduled nodes:",
            source_summary[
                "next_nodes"
            ],
        )


        graph.invoke(
            None,
            config=source_config,
            durability="sync",
        )


        # ====================================================
        # Read latest state AFTER replay
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
            new_head_summary.get(
                "checkpoint_id"
            )
        )


        print_snapshot_summary(
            "NEW THREAD HEAD AFTER REPLAY",
            new_head_summary,
        )


        # ====================================================
        # Defensive proof:
        #
        # replay should create a new checkpoint branch.
        # ====================================================

        if (
            new_head_checkpoint_id
            ==
            previous_head_checkpoint_id
        ):

            raise RuntimeError(
                (
                    "Replay completed without creating "
                    "a new thread head. "
                    "Time-travel branch was not observed."
                )
            )


        # ====================================================
        # Verify ORIGINAL SOURCE checkpoint still exists
        # ====================================================

        source_preserved = (
            exact_checkpoint_exists(
                resources,
                source_config,
            )
        )


        # ====================================================
        # Verify ORIGINAL FINAL/HEAD checkpoint still exists
        # ====================================================

        previous_head_preserved = False


        if (
            previous_head_checkpoint_id
            is not None
        ):

            old_head_config = (
                build_exact_config(
                    thread_id=
                        thread_id,

                    checkpoint_id=
                        previous_head_checkpoint_id,
                )
            )


            previous_head_preserved = (
                exact_checkpoint_exists(
                    resources,
                    old_head_config,
                )
            )


        # ====================================================
        # Export full new branch head
        # ====================================================

        new_head_full = (
            snapshot_to_full_record(
                new_head_snapshot
            )
        )


        # ====================================================
        # Replay report
        # ====================================================

        replay_report = {

            "thread_id":
                thread_id,

            "database_path":
                str(
                    resources.db_path
                ),

            "source_checkpoint": {
                **source_summary,
            },

            "previous_head": {
                **previous_head_summary,
            },

            "new_head": {
                **new_head_summary,
            },

            "source_checkpoint_preserved":
                source_preserved,

            "previous_head_preserved":
                previous_head_preserved,

            "new_head_created":
                (
                    new_head_checkpoint_id
                    !=
                    previous_head_checkpoint_id
                ),

            "interpretation": (
                "A new execution branch was created from "
                "the selected historical checkpoint. "
                "Earlier checkpoints remain addressable "
                "by checkpoint_id."
            ),
        }


        # ====================================================
        # Save replay audit
        # ====================================================

        run_dir = (
            get_thread_run_dir(
                thread_id
            )
        )


        replay_dir = (
            run_dir
            /
            "replays"
        )


        replay_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        source_short = (
            str(
                checkpoint_id
            )[
                :12
            ]
        )


        new_short = (
            str(
                new_head_checkpoint_id
            )[
                :12
            ]
        )


        replay_report_path = (
            replay_dir
            /
            (
                f"replay_"
                f"{source_short}_to_"
                f"{new_short}.json"
            )
        )


        new_state_path = (
            replay_dir
            /
            (
                f"new_head_"
                f"{new_short}.json"
            )
        )


        save_json(
            replay_report_path,
            replay_report,
        )


        save_json(
            new_state_path,
            new_head_full,
        )


        # ====================================================
        # Final verification
        # ====================================================

        print()

        print(
            "=" * 70
        )

        print(
            "TIME-TRAVEL VERIFICATION"
        )

        print(
            "=" * 70
        )


        print(
            "Source checkpoint preserved:",
            source_preserved,
        )


        print(
            "Original head preserved:",
            previous_head_preserved,
        )


        print(
            "New head created:",
            (
                new_head_checkpoint_id
                !=
                previous_head_checkpoint_id
            ),
        )


        print(
            "Replay report:",
            replay_report_path,
        )


        print(
            "New head snapshot:",
            new_state_path,
        )


        # ====================================================
        # Fail closed if history was not preserved
        # ====================================================

        if not source_preserved:

            raise RuntimeError(
                (
                    "Historical source checkpoint disappeared "
                    "after replay. This violates our "
                    "non-destructive replay requirement."
                )
            )


        if not previous_head_preserved:

            raise RuntimeError(
                (
                    "Original thread head disappeared after "
                    "replay. This violates our audit-history "
                    "requirement."
                )
            )


        print()

        print(
            "STEP 11D-B TIME TRAVEL PASSED."
        )


    finally:

        resources.close()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build replay command-line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Step 11D-B historical LangGraph replay"
        )
    )


    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )


    replay_parser = (
        subparsers.add_parser(
            "replay",
            help=(
                "Replay execution from an exact "
                "historical checkpoint."
            ),
        )
    )


    replay_parser.add_argument(
        "thread_id",
        help=(
            "Persistent LangGraph thread ID."
        ),
    )


    replay_parser.add_argument(
        "checkpoint_id",
        help=(
            "REAL historical checkpoint ID."
        ),
    )


    replay_parser.add_argument(
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
        "replay"
    ):

        command_replay(

            thread_id=
                args.thread_id,

            checkpoint_id=
                args.checkpoint_id,

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
