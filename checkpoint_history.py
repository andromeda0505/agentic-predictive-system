from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from checkpointing import (
    create_checkpoint_resources,
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
# STEP 11D
#
# CHECKPOINT HISTORY + EXACT HISTORICAL INSPECTION
#
# Commands:
#
#   python checkpoint_history.py history THREAD_ID
#
#   python checkpoint_history.py inspect \
#       THREAD_ID CHECKPOINT_ID
#
# ============================================================


# ============================================================
# Generic save helper
# ============================================================

def save_json(
    path: Path,
    value: Any,
) -> None:
    """
    Save JSON in human-readable form.
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


# ============================================================
# Snapshot helpers
# ============================================================

def get_checkpoint_id(
    snapshot,
) -> str | None:
    """
    Extract checkpoint_id from a LangGraph StateSnapshot.
    """

    config = (
        getattr(
            snapshot,
            "config",
            {},
        )
        or
        {}
    )

    configurable = (
        config.get(
            "configurable",
            {},
        )
        or
        {}
    )

    checkpoint_id = (
        configurable.get(
            "checkpoint_id"
        )
    )

    if (
        checkpoint_id
        is None
    ):
        return None

    return str(
        checkpoint_id
    )


def get_checkpoint_namespace(
    snapshot,
) -> str:
    """
    Extract checkpoint namespace.

    Root graphs normally use the empty namespace:

        ""
    """

    config = (
        getattr(
            snapshot,
            "config",
            {},
        )
        or
        {}
    )

    configurable = (
        config.get(
            "configurable",
            {},
        )
        or
        {}
    )

    return str(
        configurable.get(
            "checkpoint_ns",
            "",
        )
        or
        ""
    )


def get_parent_checkpoint_id(
    snapshot,
) -> str | None:
    """
    Extract parent checkpoint ID when available.
    """

    parent_config = (
        getattr(
            snapshot,
            "parent_config",
            None,
        )
        or
        {}
    )

    configurable = (
        parent_config.get(
            "configurable",
            {},
        )
        or
        {}
    )

    checkpoint_id = (
        configurable.get(
            "checkpoint_id"
        )
    )

    if (
        checkpoint_id
        is None
    ):
        return None

    return str(
        checkpoint_id
    )


# ============================================================
# Compact history record
# ============================================================

def snapshot_to_record(
    snapshot,
    *,
    index: int,
) -> dict[str, Any]:
    """
    Convert StateSnapshot into an audit-friendly record.

    History index 1 is normally the newest checkpoint because
    LangGraph checkpoint histories are returned newest-first.
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

        "history_index":
            index,

        "checkpoint_id":
            get_checkpoint_id(
                snapshot
            ),

        "checkpoint_ns":
            get_checkpoint_namespace(
                snapshot
            ),

        "parent_checkpoint_id":
            get_parent_checkpoint_id(
                snapshot
            ),

        "created_at":
            getattr(
                snapshot,
                "created_at",
                None,
            ),

        "status":
            (
                "complete"
                if not next_nodes
                else
                "pending"
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
# Complete historical snapshot export
# ============================================================

def snapshot_to_full_record(
    snapshot,
) -> dict[str, Any]:
    """
    Export the complete historical StateSnapshot.
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

    config = (
        getattr(
            snapshot,
            "config",
            {},
        )
        or
        {}
    )

    parent_config = (
        getattr(
            snapshot,
            "parent_config",
            None,
        )
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

        "checkpoint_ns":
            get_checkpoint_namespace(
                snapshot
            ),

        "parent_checkpoint_id":
            get_parent_checkpoint_id(
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

        "config":
            config,

        "parent_config":
            parent_config,

        "metadata":
            getattr(
                snapshot,
                "metadata",
                None,
            ),

        "values":
            values,
    }


# ============================================================
# Console history printer
# ============================================================

def print_history_record(
    record: dict[str, Any],
) -> None:
    """
    Print one checkpoint record.
    """

    print()
    print(
        "-" * 70
    )

    print(
        "History index:",
        record.get(
            "history_index"
        ),
    )

    print(
        "Checkpoint ID:",
        record.get(
            "checkpoint_id"
        ),
    )

    print(
        "Checkpoint namespace:",
        repr(
            record.get(
                "checkpoint_ns"
            )
        ),
    )

    print(
        "Created:",
        record.get(
            "created_at"
        ),
    )

    print(
        "Status:",
        record.get(
            "status"
        ),
    )

    print(
        "Next nodes:",
        record.get(
            "next_nodes"
        ),
    )

    print(
        "Artifacts:",
        record.get(
            "artifact_names"
        ),
    )

    print(
        "Revision count:",
        record.get(
            "revision_count"
        ),
    )

    print(
        "Parent checkpoint:",
        record.get(
            "parent_checkpoint_id"
        ),
    )


# ============================================================
# HISTORY
# ============================================================

def command_history(
    *,
    thread_id: str,
    database_path: Path,
    limit: int,
) -> None:
    """
    List historical checkpoints belonging to one durable
    LangGraph thread.
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
        # Thread must exist
        # ====================================================

        if not checkpoint_exists(
            resources,
            config,
        ):

            raise RuntimeError(
                (
                    f"No checkpoint history exists "
                    f"for thread '{thread_id}'."
                )
            )

        print()
        print(
            "=" * 70
        )

        print(
            "CHECKPOINT HISTORY"
        )

        print(
            "=" * 70
        )

        print(
            "Thread ID:",
            thread_id,
        )

        print(
            "Database:",
            resources.db_path,
        )

        print(
            "Limit:",
            limit,
        )

        # ====================================================
        # Retrieve history
        # ====================================================

        snapshots = list(
            graph.get_state_history(
                config,
                limit=limit,
            )
        )

        records: list[
            dict[str, Any]
        ] = []

        for (
            index,
            snapshot,
        ) in enumerate(
            snapshots,
            start=1,
        ):

            record = (
                snapshot_to_record(
                    snapshot,
                    index=index,
                )
            )

            records.append(
                record
            )

            print_history_record(
                record
            )

        # ====================================================
        # Save history
        # ====================================================

        run_dir = (
            get_thread_run_dir(
                thread_id
            )
        )

        output_path = (
            run_dir
            /
            "checkpoint_history.json"
        )

        save_json(
            output_path,
            {
                "thread_id":
                    thread_id,

                "database_path":
                    str(
                        resources.db_path
                    ),

                "checkpoint_count_returned":
                    len(
                        records
                    ),

                "history":
                    records,
            },
        )

        print()
        print(
            "=" * 70
        )

        print(
            "History entries returned:",
            len(
                records
            ),
        )

        print(
            "Saved:",
            output_path,
        )

    finally:

        resources.close()


# ============================================================
# EXACT CHECKPOINT CONFIG
# ============================================================

def build_exact_checkpoint_config(
    *,
    thread_id: str,
    checkpoint_id: str,
    checkpoint_ns: str = "",
) -> dict[str, Any]:
    """
    Build configuration selecting ONE exact checkpoint.

    This differs from build_thread_config(), which identifies
    the thread and therefore defaults to its latest checkpoint.
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
# VERIFY EXACT CHECKPOINT EXISTS
# ============================================================

def require_exact_checkpoint(
    resources,
    *,
    thread_id: str,
    checkpoint_id: str,
):
    """
    Retrieve an exact checkpoint tuple directly from SQLite.

    WHY THIS EXISTS
    ---------------

    graph.get_state() can return an empty-looking StateSnapshot
    for an unresolved historical configuration.

    That is dangerous for an audit command because:

        typo in checkpoint ID
                ↓
        empty snapshot
                ↓
        user might believe it is real history

    SqliteSaver.get_tuple(), however, gives us the exact
    database lookup contract:

        existing checkpoint -> CheckpointTuple
        missing checkpoint  -> None

    Therefore we validate against the persistence backend
    BEFORE asking the graph to materialize the state.
    """

    exact_config = (
        build_exact_checkpoint_config(
            thread_id=
                thread_id,

            checkpoint_id=
                checkpoint_id,

            checkpoint_ns="",
        )
    )

    checkpoint_tuple = (
        resources
        .checkpointer
        .get_tuple(
            exact_config
        )
    )

    if (
        checkpoint_tuple
        is None
    ):

        raise RuntimeError(
            (
                "\n"
                "EXACT CHECKPOINT NOT FOUND\n"
                "\n"
                f"Thread ID:\n"
                f"    {thread_id}\n"
                "\n"
                f"Requested checkpoint ID:\n"
                f"    {checkpoint_id}\n"
                "\n"
                "This checkpoint does not exist in the "
                "SQLite persistence database.\n"
                "\n"
                "Run:\n"
                "\n"
                f"    python checkpoint_history.py "
                f"history \"{thread_id}\"\n"
                "\n"
                "Then copy one of the REAL checkpoint IDs "
                "from that output."
            )
        )

    return checkpoint_tuple


# ============================================================
# INSPECT
# ============================================================

def command_inspect(
    *,
    thread_id: str,
    checkpoint_id: str,
    database_path: Path,
) -> None:
    """
    Inspect one EXACT historical checkpoint.

    READ ONLY.

    No graph node is executed.
    No checkpoint is changed.
    No workflow is replayed.
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

        thread_config = (
            build_thread_config(
                thread_id
            )
        )

        # ====================================================
        # Thread itself must exist
        # ====================================================

        if not checkpoint_exists(
            resources,
            thread_config,
        ):

            raise RuntimeError(
                (
                    f"Thread '{thread_id}' "
                    "does not exist."
                )
            )

        # ====================================================
        # HARD VALIDATION:
        # exact checkpoint must physically exist in SQLite
        # ====================================================

        checkpoint_tuple = (
            require_exact_checkpoint(
                resources,
                thread_id=
                    thread_id,
                checkpoint_id=
                    checkpoint_id,
            )
        )

        # ====================================================
        # Use the ACTUAL config returned by SQLite.
        #
        # This avoids reconstructing configuration details
        # unnecessarily.
        # ====================================================

        exact_config = (
            checkpoint_tuple.config
        )

        # ====================================================
        # Materialize exact historical StateSnapshot
        # ====================================================

        snapshot = (
            graph.get_state(
                exact_config
            )
        )

        actual_checkpoint_id = (
            get_checkpoint_id(
                snapshot
            )
        )

        # ====================================================
        # Second defensive verification
        # ====================================================

        if (
            actual_checkpoint_id
            !=
            checkpoint_id
        ):

            raise RuntimeError(
                (
                    "Historical checkpoint resolution "
                    "mismatch.\n\n"
                    f"Requested checkpoint:\n"
                    f"    {checkpoint_id}\n\n"
                    f"Resolved checkpoint:\n"
                    f"    {actual_checkpoint_id}\n"
                )
            )

        # ====================================================
        # Full record
        # ====================================================

        full_record = (
            snapshot_to_full_record(
                snapshot
            )
        )

        values = (
            full_record.get(
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

        # ====================================================
        # Print
        # ====================================================

        print()
        print(
            "=" * 70
        )

        print(
            "HISTORICAL CHECKPOINT INSPECTION"
        )

        print(
            "=" * 70
        )

        print(
            "Thread ID:",
            thread_id,
        )

        print(
            "Checkpoint ID:",
            actual_checkpoint_id,
        )

        print(
            "Checkpoint namespace:",
            repr(
                get_checkpoint_namespace(
                    snapshot
                )
            ),
        )

        print(
            "Created:",
            full_record.get(
                "created_at"
            ),
        )

        print(
            "Next nodes:",
            full_record.get(
                "next_nodes"
            ),
        )

        print(
            "Artifacts:",
            sorted(
                artifacts.keys()
            ),
        )

        print(
            "Revision count:",
            values.get(
                "revision_count"
            ),
        )

        print(
            "Revision limit reached:",
            values.get(
                "revision_limit_reached"
            ),
        )

        print(
            "Parent checkpoint:",
            full_record.get(
                "parent_checkpoint_id"
            ),
        )

        # ====================================================
        # Save exact snapshot
        # ====================================================

        run_dir = (
            get_thread_run_dir(
                thread_id
            )
        )

        inspection_dir = (
            run_dir
            /
            "checkpoint_inspections"
        )

        safe_checkpoint_name = (
            checkpoint_id
            .replace(
                "/",
                "_",
            )
            .replace(
                "\\",
                "_",
            )
            .replace(
                ":",
                "_",
            )
        )

        output_path = (
            inspection_dir
            /
            (
                safe_checkpoint_name
                +
                ".json"
            )
        )

        save_json(
            output_path,
            full_record,
        )

        print()
        print(
            "Saved historical snapshot:"
        )

        print(
            output_path
        )

    finally:

        resources.close()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build Step-11D command line interface.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Step 11D LangGraph checkpoint "
            "history and exact inspection"
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    # ========================================================
    # HISTORY
    # ========================================================

    history_parser = (
        subparsers.add_parser(
            "history",
            help=(
                "List checkpoint history."
            ),
        )
    )

    history_parser.add_argument(
        "thread_id",
        help=(
            "Persistent LangGraph thread ID."
        ),
    )

    history_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help=(
            "Maximum checkpoints to return."
        ),
    )

    history_parser.add_argument(
        "--db",
        default=str(
            CHECKPOINT_DB_PATH
        ),
        help=(
            "SQLite checkpoint database path."
        ),
    )

    # ========================================================
    # INSPECT
    # ========================================================

    inspect_parser = (
        subparsers.add_parser(
            "inspect",
            help=(
                "Inspect one exact historical checkpoint."
            ),
        )
    )

    inspect_parser.add_argument(
        "thread_id",
        help=(
            "Persistent LangGraph thread ID."
        ),
    )

    inspect_parser.add_argument(
        "checkpoint_id",
        help=(
            "REAL checkpoint ID returned by history."
        ),
    )

    inspect_parser.add_argument(
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
# Main
# ============================================================

def main() -> None:
    """
    Dispatch history / inspect.
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
    # HISTORY
    # ========================================================

    if (
        args.command
        ==
        "history"
    ):

        if (
            args.limit
            <=
            0
        ):

            parser.error(
                "--limit must be greater than zero."
            )

        command_history(

            thread_id=
                args.thread_id,

            database_path=
                database_path,

            limit=
                args.limit,
        )

    # ========================================================
    # INSPECT
    # ========================================================

    elif (
        args.command
        ==
        "inspect"
    ):

        command_inspect(

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