from pathlib import Path
from typing import Any
import json

from ollama import chat

from schemas.state import (
    AgentArtifact,
)

from run_step4 import (
    load_skill,
    build_agent_message,
    clean_output,
)

from tools.registry import (
    bind_tools,
)


ROOT = Path(__file__).parent

MAX_OPTIONAL_TOOL_ROUNDS = 8


# ============================================================
# Execute one Python tool
# ============================================================

def execute_tool(
    function,
    tool_name: str,
    arguments: dict,
    node_name: str,
):

    try:

        result = function(
            **arguments
        )

        if isinstance(
            result,
            str,
        ):

            result_text = result

        else:

            result_text = json.dumps(
                result,
                indent=2,
                default=str,
            )

        success = True

    except Exception as exc:

        result_text = (
            f"Tool execution failed: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        success = False


    log_entry = {

        "node":
            node_name,

        "tool":
            tool_name,

        "arguments":
            arguments,

        "result":
            result_text,

        "success":
            success,
    }


    return (
        result_text,
        log_entry,
    )


# ============================================================
# Run mandatory tools
# ============================================================

def run_mandatory_tools(
    node,
    node_name: str,
    state: dict[str, Any],
):

    if not node.mandatory_tools:

        return (
            "",
            [],
        )


    functions = bind_tools(

        node.mandatory_tools,

        state,
    )


    tool_log = []

    evidence_sections = []


    print()
    print("-" * 70)
    print("MANDATORY EVIDENCE COLLECTION")
    print("-" * 70)


    for function in functions:

        tool_name = (
            function.__name__
        )


        print()
        print(
            f"MANDATORY TOOL: "
            f"{tool_name}"
        )


        result_text, log_entry = (
            execute_tool(

                function=function,

                tool_name=tool_name,

                arguments={},

                node_name=node_name,
            )
        )


        print(
            result_text[:1000]
        )


        tool_log.append(
            log_entry
        )


        evidence_sections.append(

            f"## TOOL: {tool_name}\n\n"
            f"{result_text}"
        )


    evidence_pack = (
        "\n\n".join(
            evidence_sections
        )
    )


    return (
        evidence_pack,
        tool_log,
    )


# ============================================================
# Main node runtime
# ============================================================

def run_node_with_tools(
    workflow,
    node_name: str,
    state: dict[str, Any],
):

    node = workflow.nodes[
        node_name
    ]


    # ========================================================
    # Load Markdown skill
    # ========================================================

    metadata, instructions = (
        load_skill(

            ROOT
            /
            node.skill
        )
    )


    # ========================================================
    # Original workflow context
    # ========================================================

    base_message = (
        build_agent_message(

            workflow,

            node_name,

            state,
        )
    )


    # ========================================================
    # Mandatory deterministic tools
    # ========================================================

    (
        evidence_pack,
        mandatory_log,
    ) = run_mandatory_tools(

        node=node,

        node_name=node_name,

        state=state,
    )


    # ========================================================
    # Build final user message
    # ========================================================

    if evidence_pack:

        message = f"""
{base_message}


# MANDATORY COMPUTED EVIDENCE

The following evidence was calculated directly by
authorized Python tools before you were called.

It is authoritative empirical evidence.

Do not contradict it.

{evidence_pack}


# EVIDENCE DISCIPLINE

Do not state an empirical fact unless it is supported by:

1. authoritative dataset facts;
2. the feature contract; or
3. computed tool evidence.

If evidence is unavailable, say that it has not yet
been established.
""".strip()

    else:

        message = (
            base_message
        )


    # ========================================================
    # Optional tools
    # ========================================================

    optional_functions = bind_tools(

        node.tools,

        state,
    )


    optional_tool_map = {

        function.__name__:
            function

        for function
        in optional_functions
    }


    # ========================================================
    # Messages
    # ========================================================

    messages = [

        {
            "role":
                "system",

            "content":
                instructions,
        },

        {
            "role":
                "user",

            "content":
                message,
        },
    ]


    # Start audit log with mandatory tools.

    tool_log = list(
        mandatory_log
    )


    # ========================================================
    # Console information
    # ========================================================

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

    print(
        f"MANDATORY TOOLS: "
        f"{node.mandatory_tools}"
    )

    print(
        f"OPTIONAL TOOLS: "
        f"{node.tools}"
    )

    print("=" * 70)


    # ========================================================
    # No optional tools
    # ========================================================

    if not optional_functions:

        response = chat(

            model=metadata[
                "model"
            ],

            messages=messages,

            think=False,

            stream=False,

            options={
                "temperature":
                    0.1,
            },
        )


        artifact = AgentArtifact(

            agent=metadata[
                "name"
            ],

            skill_version=str(
                metadata[
                    "version"
                ]
            ),

            content=clean_output(
                response
                .message
                .content
                or ""
            ),
        )


        return (
            artifact,
            tool_log,
        )


    # ========================================================
    # Optional agent-tool loop
    # ========================================================

    for _ in range(
        MAX_OPTIONAL_TOOL_ROUNDS
    ):

        response = chat(

            model=metadata[
                "model"
            ],

            messages=messages,

            tools=optional_functions,

            think=False,

            stream=False,

            options={
                "temperature":
                    0.1,
            },
        )


        assistant_message = (
            response.message
        )


        messages.append(
            assistant_message
        )


        requested_calls = (

            assistant_message
            .tool_calls

            or []
        )


        # ----------------------------------------------------
        # No optional call → final artifact
        # ----------------------------------------------------

        if not requested_calls:

            artifact = (
                AgentArtifact(

                    agent=metadata[
                        "name"
                    ],

                    skill_version=str(
                        metadata[
                            "version"
                        ]
                    ),

                    content=clean_output(
                        assistant_message
                        .content
                        or ""
                    ),
                )
            )


            return (
                artifact,
                tool_log,
            )


        # ----------------------------------------------------
        # Execute optional calls
        # ----------------------------------------------------

        for tool_call in requested_calls:

            tool_name = (
                tool_call
                .function
                .name
            )


            arguments = dict(

                tool_call
                .function
                .arguments

                or {}
            )


            print()
            print(
                f"OPTIONAL TOOL CALL: "
                f"{tool_name}"
            )

            print(
                f"ARGUMENTS: "
                f"{arguments}"
            )


            function = (
                optional_tool_map
                .get(
                    tool_name
                )
            )


            if function is None:

                result_text = (
                    f"Tool '{tool_name}' "
                    "is not authorized."
                )


                log_entry = {

                    "node":
                        node_name,

                    "tool":
                        tool_name,

                    "arguments":
                        arguments,

                    "result":
                        result_text,

                    "success":
                        False,
                }


            else:

                (
                    result_text,
                    log_entry,
                ) = execute_tool(

                    function=
                        function,

                    tool_name=
                        tool_name,

                    arguments=
                        arguments,

                    node_name=
                        node_name,
                )


            print(
                "TOOL RESULT:"
            )

            print(
                result_text[:1000]
            )


            tool_log.append(
                log_entry
            )


            messages.append({

                "role":
                    "tool",

                "content":
                    result_text,

                "tool_name":
                    tool_name,
            })


    # ========================================================
    # Optional-call limit reached
    # ========================================================

    messages.append({

        "role":
            "user",

        "content": (
            "The optional tool-call limit "
            "has been reached. Produce the "
            "final report now using only "
            "the evidence already available. "
            "Do not request additional tools."
        ),
    })


    response = chat(

        model=metadata[
            "model"
        ],

        messages=messages,

        think=False,

        stream=False,

        options={
            "temperature":
                0.1,
        },
    )


    artifact = AgentArtifact(

        agent=metadata[
            "name"
        ],

        skill_version=str(
            metadata[
                "version"
            ]
        ),

        content=clean_output(
            response
            .message
            .content
            or ""
        ),
    )


    return (
        artifact,
        tool_log,
    )