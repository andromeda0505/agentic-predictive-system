from pathlib import Path
import json

from ollama import chat

from schemas.output_registry import (
    get_output_model,
)

from schemas.output_validation import (
    validate_structured_output,
)

from schemas.state import (
    AgentArtifact,
)

from run_step4 import (
    load_skill,
    build_agent_message,
)

from tools.registry import (
    bind_tools,
)

from tool_runtime import (
    run_mandatory_tools,
    execute_tool,
)


ROOT = Path(__file__).parent

MAX_OPTIONAL_TOOL_ROUNDS = 8

MAX_SCHEMA_RETRIES = 2


def run_node_structured(
    workflow,
    node_name: str,
    state: dict,
):

    node = workflow.nodes[
        node_name
    ]


    # ========================================================
    # Load skill
    # ========================================================

    metadata, instructions = (
        load_skill(
            ROOT
            /
            node.skill
        )
    )


    # ========================================================
    # Build normal context
    # ========================================================

    base_message = (
        build_agent_message(
            workflow,
            node_name,
            state,
        )
    )


    # ========================================================
    # Mandatory deterministic evidence
    # ========================================================

    (
        evidence_pack,
        mandatory_log,
    ) = run_mandatory_tools(

        node=node,

        node_name=node_name,

        state=state,
    )


    if evidence_pack:

        user_message = f"""
{base_message}


# MANDATORY COMPUTED EVIDENCE

The following results were computed directly by
authorized deterministic Python tools.

They are authoritative.

{evidence_pack}


# FINAL EVIDENCE RULE

Do not copy a numerical fact from memory.

Use only the task, authoritative state,
feature contract, and tool evidence.

The final structured response will be
validated against deterministic state.
""".strip()

    else:

        user_message = (
            base_message
        )


    messages = [

        {
            "role": "system",
            "content":
                instructions,
        },

        {
            "role": "user",
            "content":
                user_message,
        },
    ]


    tool_log = list(
        mandatory_log
    )


    # ========================================================
    # Optional agent-selected tools
    # ========================================================

    optional_functions = bind_tools(

        node.tools,

        state,
    )


    tool_map = {

        function.__name__:
            function

        for function
        in optional_functions
    }


    for _ in range(
        MAX_OPTIONAL_TOOL_ROUNDS
    ):

        if not optional_functions:

            break


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
                    0.1
            },
        )


        assistant_message = (
            response.message
        )


        messages.append(
            assistant_message
        )


        calls = (
            assistant_message
            .tool_calls
            or []
        )


        if not calls:

            break


        for call in calls:

            tool_name = (
                call
                .function
                .name
            )


            arguments = dict(

                call
                .function
                .arguments
                or {}
            )


            function = (
                tool_map
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
    # Output schema
    # ========================================================

    if not node.output_schema:

        raise ValueError(
            f"Node '{node_name}' "
            "has no output_schema."
        )


    output_model = (
        get_output_model(
            node.output_schema
        )
    )


    json_schema = (
        output_model
        .model_json_schema()
    )


    messages.append({

        "role":
            "user",

        "content":
            (
                "Produce the FINAL response now. "
                "Return only the structured object "
                "required by the supplied JSON schema. "
                "Do not include Markdown, commentary, "
                "code fences, or additional fields."
            ),
    })


    # ========================================================
    # Structured-output validation loop
    # ========================================================

    last_error = None


    for attempt in range(
        MAX_SCHEMA_RETRIES + 1
    ):

        response = chat(

            model=metadata[
                "model"
            ],

            messages=messages,

            format=json_schema,

            think=False,

            stream=False,

            options={
                "temperature":
                    0,
            },
        )


        raw = (
            response
            .message
            .content
            or ""
        )


        try:

            parsed = (
                output_model
                .model_validate_json(
                    raw
                )
            )


            validate_structured_output(

                node_name=
                    node_name,

                parsed_output=
                    parsed,

                state=
                    state,

                node_spec=
                    node,
            )


            structured_data = (
                parsed.model_dump()
            )


            content = json.dumps(

                structured_data,

                indent=2,

                default=str,
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

                schema_name=
                    node.output_schema,

                structured_data=
                    structured_data,

                content=
                    content,
            )


            return (
                artifact,
                tool_log,
            )


        except Exception as exc:

            last_error = exc


            if (
                attempt
                >=
                MAX_SCHEMA_RETRIES
            ):

                break


            messages.append({

                "role":
                    "assistant",

                "content":
                    raw,
            })


            messages.append({

                "role":
                    "user",

                "content": (
                    "Your previous structured response "
                    "failed validation.\n\n"
                    f"{exc}\n\n"
                    "Correct the response using only "
                    "the authoritative evidence already "
                    "provided. Return only the corrected "
                    "structured object."
                ),
            })


    raise RuntimeError(

        f"Structured output for "
        f"node '{node_name}' "
        f"failed validation after "
        f"{MAX_SCHEMA_RETRIES + 1} attempts.\n"
        f"Last error:\n{last_error}"
    )
