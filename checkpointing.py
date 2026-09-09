from __future__ import annotations

import sqlite3

from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)


# ============================================================
# Checkpoint resource bundle
# ============================================================

@dataclass
class CheckpointResources:
    """
    Own the resources needed for durable LangGraph
    checkpointing.

    Why bundle these together?

    SqliteSaver uses a live SQLite connection.

    Therefore we need to keep BOTH:

        checkpointer
        connection

    alive for the lifetime of the graph execution.
    """

    db_path: Path

    connection: sqlite3.Connection

    checkpointer: SqliteSaver


    def close(
        self,
    ) -> None:
        """
        Close the SQLite connection cleanly.

        Safe to call more than once.
        """

        try:

            self.connection.close()

        except sqlite3.Error:

            pass


# ============================================================
# Create SQLite checkpoint backend
# ============================================================

def create_checkpoint_resources(
    db_path: str | Path,
) -> CheckpointResources:
    """
    Create a persistent SQLite-backed LangGraph checkpointer.

    Parameters
    ----------
    db_path:
        Location of the SQLite checkpoint database.

        Example:

            runs/checkpoints/workflow.sqlite

    Returns
    -------
    CheckpointResources

        Contains:

            db_path
            connection
            checkpointer

    Important
    ---------
    check_same_thread=False is intentional.

    Our LangGraph contains parallel branches:

        researcher
        statistician

    LangGraph may execute those branches using different
    worker threads.

    The default Python SQLite setting restricts a connection
    to the thread where it was created.

    Therefore we explicitly allow LangGraph's worker threads
    to use this connection.
    """

    path = Path(
        db_path
    ).expanduser().resolve()


    # ========================================================
    # Create checkpoint directory
    # ========================================================

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Open durable SQLite database
    # ========================================================

    connection = sqlite3.connect(
        str(
            path
        ),
        check_same_thread=False,
    )


    # ========================================================
    # Basic SQLite reliability settings
    # ========================================================

    connection.execute(
        "PRAGMA busy_timeout = 5000;"
    )


    # ========================================================
    # Construct LangGraph saver
    # ========================================================

    checkpointer = SqliteSaver(
        connection
    )


    # ========================================================
    # Initialize checkpoint tables when supported
    #
    # Current SqliteSaver implementations expose setup().
    # hasattr keeps this wrapper tolerant of API variants.
    # ========================================================

    if hasattr(
        checkpointer,
        "setup",
    ):

        checkpointer.setup()


    return CheckpointResources(

        db_path=path,

        connection=connection,

        checkpointer=checkpointer,
    )


# ============================================================
# Checkpoint database inspection
# ============================================================

def list_checkpoint_tables(
    connection: sqlite3.Connection,
) -> list[str]:
    """
    Return SQLite table names.

    This is useful for proving that LangGraph's checkpoint
    backend actually initialized persistent storage.
    """

    cursor = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    )


    return [
        str(
            row[0]
        )

        for row
        in cursor.fetchall()
    ]


# ============================================================
# Convenience diagnostic
# ============================================================

def describe_checkpoint_resources(
    resources: CheckpointResources,
) -> dict:
    """
    Return a small diagnostic description.

    This contains no workflow state yet.

    Step 11A only proves that durable persistence storage
    exists and can be initialized.
    """

    return {

        "database_path":
            str(
                resources.db_path
            ),

        "database_exists":
            resources.db_path.exists(),

        "checkpointer_type":
            type(
                resources.checkpointer
            ).__name__,

        "tables":
            list_checkpoint_tables(
                resources.connection
            ),
    }
