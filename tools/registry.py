from typing import Any, Callable

from tools.data_tools import (
    build_data_tools,
)


def bind_tools(
    tool_names: list[str],
    state: dict[str, Any],
) -> list[Callable]:

    """
    Bind declared tool names to actual Python
    functions for the current workflow state.
    """

    available = (
        build_data_tools(
            state
        )
    )


    unknown = [

        name

        for name
        in tool_names

        if name not in available
    ]


    if unknown:

        raise ValueError(
            "Unknown tools declared "
            f"in workflow: {unknown}"
        )


    return [

        available[
            name
        ]

        for name
        in tool_names
    ]
