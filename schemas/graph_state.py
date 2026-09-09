from operator import add
from typing import Annotated, Any

from typing_extensions import TypedDict


def merge_dicts(
    left: dict,
    right: dict,
) -> dict:

    result = dict(left)

    result.update(
        right
    )

    return result


class GraphState(TypedDict):

    task: str

    dataset_path: str

    dataset_facts: dict[
        str,
        Any,
    ]

    feature_policy: dict[
        str,
        Any,
    ]

    artifacts: Annotated[
        dict[str, Any],
        merge_dicts,
    ]

    warnings: Annotated[
        list[str],
        add,
    ]

    tool_log: Annotated[
        list[dict[str, Any]],
        add,
    ]

    # ========================================================
    # Step 9 revision control
    # ========================================================

    revision_count: int

    max_revisions: int

    revision_feedback: dict[
        str,
        Any,
    ]

    revision_limit_reached: bool