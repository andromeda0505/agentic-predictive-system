from __future__ import annotations

import re
from typing import Any

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from schemas.graph_state import GraphState

from schemas.unstructured_validation import (
    build_researcher_safe_fallback,
    researcher_retry_feedback,
    validate_unstructured_output,
)

from structured_runtime import (
    run_node_structured,
)

from tool_runtime import (
    run_node_with_tools,
)


# ============================================================
# Configuration
# ============================================================

MAX_UNSTRUCTURED_VALIDATION_ATTEMPTS = 3

# run_node_structured() already performs internal retries.
#
# This provides one additional feedback-informed outer cycle.
MAX_STRUCTURED_FEEDBACK_ROUNDS = 2


# ============================================================
# Route-path resolution
# ============================================================

def _resolve_route_path(
    state: dict[str, Any],
    path: str,
) -> Any:
    """
    Resolve a dot-separated path from LangGraph state.

    Example:

        artifacts.critic.structured_data.recommended_next_action

    Whitespace is stripped because YAML folded strings may
    contain trailing newlines.
    """

    clean_path = str(
        path
    ).strip()


    if not clean_path:

        raise ValueError(
            "Conditional route path is empty."
        )


    current: Any = state


    for raw_part in clean_path.split(
        "."
    ):

        part = raw_part.strip()


        if not part:

            raise ValueError(
                (
                    "Invalid empty component in route path "
                    f"'{clean_path}'."
                )
            )


        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if isinstance(
            current,
            dict,
        ):

            if (
                part
                not in current
            ):

                raise KeyError(
                    (
                        "Could not resolve workflow route "
                        f"path '{clean_path}'. "
                        f"Missing key '{part}'."
                    )
                )


            current = current[
                part
            ]


        # ----------------------------------------------------
        # Object / Pydantic model
        # ----------------------------------------------------

        else:

            if not hasattr(
                current,
                part,
            ):

                raise AttributeError(
                    (
                        "Could not resolve workflow route "
                        f"path '{clean_path}'. "
                        f"Object of type "
                        f"'{type(current).__name__}' "
                        f"has no attribute '{part}'."
                    )
                )


            current = getattr(
                current,
                part,
            )


    return current


# ============================================================
# Conditional router
# ============================================================

def _make_conditional_router(
    node_name: str,
    node_spec,
):
    """
    Create the routing function for one conditional node.
    """

    def router(
        state: dict[str, Any],
    ) -> str:

        # ----------------------------------------------------
        # Hard revision-budget stop
        # ----------------------------------------------------

        if state.get(
            "revision_limit_reached",
            False,
        ):

            print()

            print(
                "REVISION LIMIT REACHED"
            )

            print(
                "Routing workflow to END."
            )


            return (
                "__revision_limit__"
            )


        # ----------------------------------------------------
        # Resolve route value from state
        # ----------------------------------------------------

        route_value = (
            _resolve_route_path(
                state,
                node_spec.route_on,
            )
        )


        route_value = str(
            route_value
        ).strip()


        # ----------------------------------------------------
        # Validate against YAML declaration
        # ----------------------------------------------------

        if (
            route_value
            not in node_spec.routes
        ):

            raise ValueError(
                (
                    f"Conditional node '{node_name}' "
                    f"produced undeclared route "
                    f"'{route_value}'. "
                    "Declared routes: "
                    f"{sorted(node_spec.routes.keys())}"
                )
            )


        print()

        print(
            (
                "CONDITIONAL ROUTE: "
                f"{node_name} -> {route_value}"
            )
        )


        return route_value


    return router


# ============================================================
# Parallel-group validation
# ============================================================

def _validate_parallel_groups(
    workflow,
):
    """
    Validate declarative fan-out / fan-in structures.

    Example:

                      researcher
                     /          \\
        planner -----            ----- critic
                     \\          /
                      statistician
    """

    node_names = set(
        workflow.nodes.keys()
    )


    ordinary_edges = {
        (
            edge.from_node,
            edge.to_node,
        )

        for edge
        in workflow.edges
    }


    # --------------------------------------------------------
    # Gather conditional destinations
    # --------------------------------------------------------

    conditional_destinations: list[
        tuple[
            str,
            str,
            str,
        ]
    ] = []


    for (
        conditional_node_name,
        conditional_node_spec,
    ) in workflow.nodes.items():

        for (
            route_label,
            destination,
        ) in conditional_node_spec.routes.items():

            destination = str(
                destination
            ).strip()


            if (
                destination
                !=
                "__end__"
            ):

                conditional_destinations.append(
                    (
                        conditional_node_name,
                        str(
                            route_label
                        ).strip(),
                        destination,
                    )
                )


    seen_sources: set[
        str
    ] = set()


    seen_branch_memberships: dict[
        str,
        str,
    ] = {}


    # --------------------------------------------------------
    # Validate every parallel group
    # --------------------------------------------------------

    for (
        group_index,
        group,
    ) in enumerate(
        workflow.parallel_groups,
        start=1,
    ):

        source = str(
            group.from_node
        ).strip()


        branches = [

            str(
                branch
            ).strip()

            for branch
            in group.branches
        ]


        join = str(
            group.join
        ).strip()


        # ----------------------------------------------------
        # Source exists
        # ----------------------------------------------------

        if (
            source
            not in node_names
        ):

            raise ValueError(
                (
                    f"Parallel group {group_index} "
                    f"references unknown source "
                    f"'{source}'."
                )
            )


        # ----------------------------------------------------
        # Join exists
        # ----------------------------------------------------

        if (
            join
            not in node_names
        ):

            raise ValueError(
                (
                    f"Parallel group {group_index} "
                    f"references unknown join "
                    f"'{join}'."
                )
            )


        # ----------------------------------------------------
        # Require real fan-out
        # ----------------------------------------------------

        if (
            len(
                branches
            )
            <
            2
        ):

            raise ValueError(
                (
                    f"Parallel group {group_index} "
                    "must contain at least two branches."
                )
            )


        # ----------------------------------------------------
        # Unique branch names
        # ----------------------------------------------------

        if (
            len(
                branches
            )
            !=
            len(
                set(
                    branches
                )
            )
        ):

            raise ValueError(
                (
                    f"Parallel group {group_index} "
                    "contains duplicate branches."
                )
            )


        # ----------------------------------------------------
        # All branches exist
        # ----------------------------------------------------

        unknown_branches = sorted(

            branch

            for branch
            in branches

            if (
                branch
                not in node_names
            )
        )


        if unknown_branches:

            raise ValueError(
                (
                    f"Parallel group {group_index} "
                    "references unknown branch nodes: "
                    f"{unknown_branches}"
                )
            )


        # ----------------------------------------------------
        # Structural sanity
        # ----------------------------------------------------

        if (
            source
            in branches
        ):

            raise ValueError(
                (
                    f"Parallel source '{source}' "
                    "cannot also be one of its branches."
                )
            )


        if (
            join
            in branches
        ):

            raise ValueError(
                (
                    f"Parallel join '{join}' "
                    "cannot also be one of its branches."
                )
            )


        if (
            source
            ==
            join
        ):

            raise ValueError(
                (
                    "Parallel source and join "
                    "must be different nodes."
                )
            )


        # ----------------------------------------------------
        # One Step-10 fan-out per source
        # ----------------------------------------------------

        if (
            source
            in seen_sources
        ):

            raise ValueError(
                (
                    f"Parallel source '{source}' "
                    "is declared more than once."
                )
            )


        seen_sources.add(
            source
        )


        # ----------------------------------------------------
        # Branch membership
        # ----------------------------------------------------

        for branch in branches:

            if (
                branch
                in seen_branch_memberships
            ):

                previous_source = (
                    seen_branch_memberships[
                        branch
                    ]
                )


                raise ValueError(
                    (
                        f"Branch '{branch}' belongs to "
                        "multiple parallel groups: "
                        f"'{previous_source}' and "
                        f"'{source}'."
                    )
                )


            seen_branch_memberships[
                branch
            ] = source


        # ----------------------------------------------------
        # parallel_groups owns source -> branch
        # ----------------------------------------------------

        for branch in branches:

            if (
                (
                    source,
                    branch,
                )
                in ordinary_edges
            ):

                raise ValueError(
                    (
                        "Ordinary edge duplicates "
                        "parallel fan-out "
                        f"'{source} -> {branch}'. "
                        "Remove the ordinary edge."
                    )
                )


        # ----------------------------------------------------
        # parallel_groups owns branch -> join barrier
        # ----------------------------------------------------

        for branch in branches:

            if (
                (
                    branch,
                    join,
                )
                in ordinary_edges
            ):

                raise ValueError(
                    (
                        "Ordinary edge duplicates "
                        "parallel join "
                        f"'{branch} -> {join}'. "
                        "Remove the ordinary edge."
                    )
                )


        # ----------------------------------------------------
        # Prevent conditional jumps directly into one branch
        # ----------------------------------------------------

        for (
            conditional_node,
            route_label,
            destination,
        ) in conditional_destinations:

            if (
                destination
                in branches
            ):

                raise ValueError(
                    (
                        f"Conditional route "
                        f"'{conditional_node} "
                        f"--{route_label}--> "
                        f"{destination}' jumps directly "
                        "into a parallel branch. "
                        "That could leave the join barrier "
                        "unsatisfied. "
                        f"Route through fan-out source "
                        f"'{source}' instead."
                    )
                )


# ============================================================
# Safe structured feedback
# ============================================================

def _safe_structured_feedback(
    node_name: str,
    error: Exception,
) -> list[str]:
    """
    Convert a structured-runtime error into sanitized feedback.

    Rejected text is intentionally not copied verbatim because
    a deterministic local model may simply echo it.
    """

    lower = str(
        error
    ).lower()


    instructions: list[
        str
    ] = []


    # --------------------------------------------------------
    # Statistician censoring contract
    # --------------------------------------------------------

    if (
        node_name
        ==
        "statistician"
        and
        "construction_end_date"
        in lower
        and
        "imput"
        in lower
    ):

        instructions.append(
            (
                "CENSORING CONTRACT: Remove every "
                "recommendation or question treating the "
                "historical target end date as an ordinary "
                "missing value. Handle unfinished homes "
                "through censoring-aware reasoning. "
                "Ordinary imputation may only be discussed "
                "for legitimate prediction-time predictors."
            )
        )


    # --------------------------------------------------------
    # Safe semantic category labels
    # --------------------------------------------------------

    categories = sorted(
        set(
            re.findall(
                r"\[([a-z_]+)\]",
                lower,
            )
        )
    )


    if categories:

        instructions.append(
            (
                "Correct all deterministic semantic "
                "validation categories: "
                +
                ", ".join(
                    categories
                )
                +
                "."
            )
        )


    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    if not instructions:

        instructions.append(
            (
                "The previous structured output failed "
                "deterministic semantic validation. "
                "Re-read authoritative evidence and remove "
                "unsupported claims rather than merely "
                "paraphrasing them."
            )
        )


    return instructions


# ============================================================
# Structured execution wrapper
# ============================================================

def _run_structured_with_feedback(
    workflow,
    node_name: str,
    state: dict[str, Any],
):
    """
    Run a structured node with an outer feedback loop.

    run_node_structured() already has internal retries.

    These local correction rounds do NOT increment the
    graph-level Critic revision counter.
    """

    runtime_state = state


    last_error: Exception | None = None


    for feedback_round in range(
        1,
        MAX_STRUCTURED_FEEDBACK_ROUNDS + 1,
    ):

        print()

        print(
            (
                "STRUCTURED FEEDBACK ROUND: "
                f"{node_name} "
                f"{feedback_round}/"
                f"{MAX_STRUCTURED_FEEDBACK_ROUNDS}"
            )
        )


        try:

            return run_node_structured(
                workflow,
                node_name,
                runtime_state,
            )


        except RuntimeError as exc:

            last_error = exc


            print()

            print(
                (
                    "STRUCTURED RUNTIME EXHAUSTED: "
                    f"{node_name}"
                )
            )


            if (
                feedback_round
                >=
                MAX_STRUCTURED_FEEDBACK_ROUNDS
            ):

                break


            safe_feedback = (
                _safe_structured_feedback(
                    node_name,
                    exc,
                )
            )


            runtime_state = dict(
                state
            )


            runtime_state[
                "revision_feedback"
            ] = {

                "feedback_type":
                    "local_structured_validation",

                "agent":
                    node_name,

                "feedback_round":
                    feedback_round,

                "local_validation_errors":
                    safe_feedback,

                "instruction":
                    (
                        "Correct every listed validation "
                        "problem. Do not repeat or "
                        "paraphrase rejected content."
                    ),

                "prior_revision_feedback":
                    state.get(
                        "revision_feedback",
                        {},
                    ),
            }


            print()

            print(
                (
                    "INJECTING SANITIZED STRUCTURED "
                    f"FEEDBACK INTO: {node_name}"
                )
            )


    raise RuntimeError(
        (
            f"Structured node '{node_name}' failed after "
            f"{MAX_STRUCTURED_FEEDBACK_ROUNDS} outer "
            "feedback rounds.\n\n"
            "Last error:\n"
            f"{last_error}"
        )
    )


# ============================================================
# Unstructured execution wrapper
# ============================================================

def _run_unstructured_with_validation(
    workflow,
    node_name: str,
    state: dict[str, Any],
):
    """
    Execute an unstructured node through:

        generation
            ↓
        deterministic validation
            ↓
        sanitized local retry
            ↓
        deterministic Researcher fallback

    Local retries do not consume graph revision_count.
    """

    runtime_state = state


    accumulated_tool_log = []


    last_artifact = None


    last_error: Exception | None = None


    for attempt in range(
        1,
        MAX_UNSTRUCTURED_VALIDATION_ATTEMPTS + 1,
    ):

        print()

        print(
            (
                "UNSTRUCTURED VALIDATION ATTEMPT: "
                f"{node_name} "
                f"{attempt}/"
                f"{MAX_UNSTRUCTURED_VALIDATION_ATTEMPTS}"
            )
        )


        (
            artifact,
            tool_log,
        ) = run_node_with_tools(
            workflow,
            node_name,
            runtime_state,
        )


        last_artifact = artifact


        accumulated_tool_log.extend(
            tool_log
        )


        try:

            validate_unstructured_output(
                node_name,
                artifact.content,
                state,
            )


            print()

            print(
                (
                    "UNSTRUCTURED SEMANTIC "
                    f"VALIDATION PASSED: {node_name}"
                )
            )


            return (
                artifact,
                accumulated_tool_log,
            )


        except ValueError as exc:

            last_error = exc


            print()

            print(
                (
                    "UNSTRUCTURED SEMANTIC "
                    f"VALIDATION FAILED: {node_name}"
                )
            )


            print(
                str(
                    exc
                )
            )


            if (
                attempt
                >=
                MAX_UNSTRUCTURED_VALIDATION_ATTEMPTS
            ):

                break


            # ------------------------------------------------
            # Sanitized feedback
            # ------------------------------------------------

            if (
                node_name
                ==
                "researcher"
            ):

                safe_feedback = (
                    researcher_retry_feedback(
                        str(
                            exc
                        )
                    )
                )


            else:

                safe_feedback = [
                    (
                        "Remove every unsupported factual "
                        "claim and use only authoritative "
                        "state."
                    )
                ]


            runtime_state = dict(
                state
            )


            runtime_state[
                "revision_feedback"
            ] = {

                "feedback_type":
                    "local_unstructured_validation",

                "agent":
                    node_name,

                "failed_attempt":
                    attempt,

                "local_validation_errors":
                    safe_feedback,

                "instruction":
                    (
                        "Correct every listed problem "
                        "without repeating rejected content."
                    ),

                "prior_revision_feedback":
                    state.get(
                        "revision_feedback",
                        {},
                    ),
            }


            print()

            print(
                (
                    "INJECTING SANITIZED UNSTRUCTURED "
                    f"FEEDBACK INTO: {node_name}"
                )
            )


    # ========================================================
    # Deterministic Researcher fallback
    # ========================================================

    if (
        node_name
        ==
        "researcher"
        and
        last_artifact
        is not None
    ):

        print()

        print(
            "=" * 70
        )

        print(
            "RESEARCHER DETERMINISTIC FALLBACK ACTIVATED"
        )

        print(
            "=" * 70
        )


        fallback_content = (
            build_researcher_safe_fallback(
                state
            )
        )


        # Validate deterministic fallback too.
        validate_unstructured_output(
            node_name,
            fallback_content,
            state,
        )


        fallback_artifact = (
            last_artifact.model_copy(
                update={
                    "content":
                        fallback_content
                }
            )
        )


        print()

        print(
            "DETERMINISTIC RESEARCHER FALLBACK VALIDATED."
        )


        return (
            fallback_artifact,
            accumulated_tool_log,
        )


    # ========================================================
    # Other unstructured agents fail closed
    # ========================================================

    raise RuntimeError(
        (
            f"Unstructured output for node '{node_name}' "
            "failed semantic validation after "
            f"{MAX_UNSTRUCTURED_VALIDATION_ATTEMPTS} "
            "attempts.\n\n"
            "Last error:\n"
            f"{last_error}"
        )
    )


# ============================================================
# Workflow compiler
# ============================================================

def compile_workflow(
    workflow,
    checkpointer: Any | None = None,
):
    """
    Compile WorkflowDefinition into executable LangGraph.

    Parameters
    ----------
    workflow:
        Declarative workflow specification.

    checkpointer:
        Optional LangGraph checkpoint saver.

        Examples:

            None
                -> ordinary non-persistent execution

            SqliteSaver
                -> durable thread-scoped persistence

            InMemorySaver
                -> temporary checkpointing for testing

    Important
    ---------
    Passing a checkpointer here only MAKES the graph capable
    of persistence.

    Actual persisted execution also requires a thread_id when
    graph.invoke() is called.

    That invocation behavior is Step 11C.
    """

    # ========================================================
    # 1. Validate graph structure
    # ========================================================

    _validate_parallel_groups(
        workflow
    )


    # ========================================================
    # 2. Create StateGraph builder
    # ========================================================

    graph = StateGraph(
        GraphState
    )


    # ========================================================
    # Node factory
    # ========================================================

    def make_node_function(
        node_name: str,
        node_spec,
    ):
        """
        Create executable node function from workflow spec.
        """

        def node_function(
            state: dict[str, Any],
        ) -> dict[str, Any]:

            print()

            print(
                "=" * 70
            )

            print(
                f"NODE: {node_name}"
            )

            print(
                "=" * 70
            )


            # =================================================
            # Structured node
            # =================================================

            if (
                node_spec.output_schema
                is not None
            ):

                (
                    artifact,
                    tool_log,
                ) = (
                    _run_structured_with_feedback(
                        workflow,
                        node_name,
                        state,
                    )
                )


            # =================================================
            # Unstructured node
            # =================================================

            else:

                (
                    artifact,
                    tool_log,
                ) = (
                    _run_unstructured_with_validation(
                        workflow,
                        node_name,
                        state,
                    )
                )


            # =================================================
            # Publish validated artifact
            # =================================================

            update: dict[
                str,
                Any,
            ] = {

                "artifacts": {
                    node_spec.output:
                        artifact.model_dump()
                },

                "tool_log":
                    tool_log,
            }


            # =================================================
            # Critic controls GRAPH-LEVEL revision state
            # =================================================

            if (
                node_name
                ==
                "critic"
            ):

                critic_data = (
                    artifact.structured_data
                    or
                    {}
                )


                action = str(
                    critic_data.get(
                        "recommended_next_action",
                        "accept",
                    )
                ).strip()


                # ---------------------------------------------
                # Commit Critic feedback to actual shared state
                # ---------------------------------------------

                update[
                    "revision_feedback"
                ] = critic_data


                current_count = int(
                    state.get(
                        "revision_count",
                        0,
                    )
                )


                max_revisions = int(
                    state.get(
                        "max_revisions",
                        0,
                    )
                )


                # ---------------------------------------------
                # Accept
                # ---------------------------------------------

                if (
                    action
                    ==
                    "accept"
                ):

                    update[
                        "revision_count"
                    ] = current_count


                    update[
                        "revision_limit_reached"
                    ] = False


                # ---------------------------------------------
                # Revision budget remains
                # ---------------------------------------------

                elif (
                    current_count
                    <
                    max_revisions
                ):

                    new_count = (
                        current_count
                        +
                        1
                    )


                    update[
                        "revision_count"
                    ] = new_count


                    update[
                        "revision_limit_reached"
                    ] = False


                    print()

                    print(
                        (
                            "REVISION ROUND: "
                            f"{new_count}/"
                            f"{max_revisions}"
                        )
                    )


                # ---------------------------------------------
                # Revision budget exhausted
                # ---------------------------------------------

                else:

                    update[
                        "revision_count"
                    ] = current_count


                    update[
                        "revision_limit_reached"
                    ] = True


                    print()

                    print(
                        (
                            "REVISION REQUESTED BUT "
                            "MAXIMUM REVISION COUNT "
                            "HAS BEEN REACHED."
                        )
                    )


            return update


        return node_function


    # ========================================================
    # 3. Register nodes
    # ========================================================

    for (
        node_name,
        node_spec,
    ) in workflow.nodes.items():

        graph.add_node(
            node_name,
            make_node_function(
                node_name,
                node_spec,
            ),
        )


    # ========================================================
    # 4. START
    # ========================================================

    if (
        workflow.start
        not in workflow.nodes
    ):

        raise ValueError(
            (
                "Configured workflow start node "
                f"'{workflow.start}' does not exist."
            )
        )


    graph.add_edge(
        START,
        workflow.start,
    )


    # ========================================================
    # 5. Ordinary sequential edges
    # ========================================================

    for edge in workflow.edges:

        from_node = str(
            edge.from_node
        ).strip()


        to_node = str(
            edge.to_node
        ).strip()


        if (
            from_node
            not in workflow.nodes
        ):

            raise ValueError(
                (
                    "Workflow edge references unknown "
                    f"source node '{from_node}'."
                )
            )


        if (
            to_node
            not in workflow.nodes
        ):

            raise ValueError(
                (
                    "Workflow edge references unknown "
                    f"destination node '{to_node}'."
                )
            )


        graph.add_edge(
            from_node,
            to_node,
        )


    # ========================================================
    # 6. Parallel fan-out / fan-in
    # ========================================================

    for group in workflow.parallel_groups:

        source = str(
            group.from_node
        ).strip()


        branches = [

            str(
                branch
            ).strip()

            for branch
            in group.branches
        ]


        join = str(
            group.join
        ).strip()


        print()

        print(
            (
                "COMPILING PARALLEL GROUP: "
                f"{source} -> "
                f"{branches} -> "
                f"{join}"
            )
        )


        # ----------------------------------------------------
        # FAN-OUT
        # ----------------------------------------------------

        for branch in branches:

            graph.add_edge(
                source,
                branch,
            )


        # ----------------------------------------------------
        # FAN-IN BARRIER
        #
        # Join executes only after all branches complete.
        # ----------------------------------------------------

        graph.add_edge(
            branches,
            join,
        )


    # ========================================================
    # 7. Conditional routing
    # ========================================================

    conditional_nodes: set[
        str
    ] = set()


    for (
        node_name,
        node_spec,
    ) in workflow.nodes.items():

        if (
            node_spec.route_on
            is None
        ):

            continue


        if not node_spec.routes:

            raise ValueError(
                (
                    f"Node '{node_name}' declares "
                    "'route_on' but has no routes."
                )
            )


        conditional_nodes.add(
            node_name
        )


        path_map: dict[
            str,
            Any,
        ] = {}


        for (
            raw_route_label,
            raw_destination,
        ) in node_spec.routes.items():

            route_label = str(
                raw_route_label
            ).strip()


            destination = str(
                raw_destination
            ).strip()


            # ------------------------------------------------
            # END
            # ------------------------------------------------

            if (
                destination
                ==
                "__end__"
            ):

                path_map[
                    route_label
                ] = END


            # ------------------------------------------------
            # Normal destination node
            # ------------------------------------------------

            else:

                if (
                    destination
                    not in workflow.nodes
                ):

                    raise ValueError(
                        (
                            f"Conditional node "
                            f"'{node_name}' route "
                            f"'{route_label}' points to "
                            "unknown destination "
                            f"'{destination}'."
                        )
                    )


                path_map[
                    route_label
                ] = destination


        # ----------------------------------------------------
        # Revision-limit escape route
        # ----------------------------------------------------

        path_map[
            "__revision_limit__"
        ] = END


        graph.add_conditional_edges(
            node_name,
            _make_conditional_router(
                node_name,
                node_spec,
            ),
            path_map,
        )


    # ========================================================
    # 8. Static END edge
    #
    # Conditional end nodes already own their exit behavior.
    # ========================================================

    if (
        workflow.end
        not in conditional_nodes
    ):

        if (
            workflow.end
            not in workflow.nodes
        ):

            raise ValueError(
                (
                    "Configured workflow end node "
                    f"'{workflow.end}' does not exist."
                )
            )


        graph.add_edge(
            workflow.end,
            END,
        )


    # ========================================================
    # 9. Compile executable graph
    #
    # STEP 11B CHANGE IS HERE.
    # ========================================================

    if (
        checkpointer
        is None
    ):

        print()

        print(
            "COMPILING WITHOUT CHECKPOINTER"
        )


        compiled_graph = (
            graph.compile()
        )


    else:

        print()

        print(
            (
                "COMPILING WITH CHECKPOINTER: "
                f"{type(checkpointer).__name__}"
            )
        )


        compiled_graph = (
            graph.compile(
                checkpointer=checkpointer,
            )
        )


    # ========================================================
    # Defensive compiler contract
    # ========================================================

    if not hasattr(
        compiled_graph,
        "invoke",
    ):

        raise TypeError(
            (
                "LangGraph compilation did not return an "
                "executable graph. "
                "Expected an object supporting .invoke(). "
                f"Received type: "
                f"{type(compiled_graph).__name__}."
            )
        )


    return compiled_graph