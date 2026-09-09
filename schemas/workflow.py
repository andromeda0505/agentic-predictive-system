from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# Context supplied to a workflow node
# ============================================================

class ContextItem(
    BaseModel
):
    """
    One state item supplied to an agent.

    Example YAML:

        - path: dataset_facts
          title: AUTHORITATIVE DATASET FACTS
    """

    path: str

    title: str


# ============================================================
# Workflow node
# ============================================================

class WorkflowNode(
    BaseModel
):
    """
    Declarative definition of one workflow node.
    """

    # --------------------------------------------------------
    # Agent skill file
    # --------------------------------------------------------

    skill: str


    # --------------------------------------------------------
    # Artifact name written into shared state
    # --------------------------------------------------------

    output: str


    # --------------------------------------------------------
    # Optional structured-output schema
    # --------------------------------------------------------

    output_schema: (
        str
        |
        None
    ) = None


    # --------------------------------------------------------
    # State context supplied to the node
    # --------------------------------------------------------

    context: list[
        ContextItem
    ]


    # --------------------------------------------------------
    # Tools that MUST run before the agent reasons
    # --------------------------------------------------------

    mandatory_tools: list[
        str
    ] = Field(
        default_factory=list
    )


    # --------------------------------------------------------
    # Optional tools available to the agent/runtime
    # --------------------------------------------------------

    tools: list[
        str
    ] = Field(
        default_factory=list
    )


    # --------------------------------------------------------
    # Node-specific assignment
    # --------------------------------------------------------

    assignment: str


    # --------------------------------------------------------
    # Optional conditional-routing configuration
    #
    # Example:
    #
    # route_on:
    #     artifacts.critic.structured_data
    #     .recommended_next_action
    #
    # routes:
    #     accept: __end__
    #     revise_both: planner
    # --------------------------------------------------------

    route_on: (
        str
        |
        None
    ) = None


    routes: dict[
        str,
        str,
    ] = Field(
        default_factory=dict
    )


# ============================================================
# Ordinary directed edge
# ============================================================

class WorkflowEdge(
    BaseModel
):
    """
    Ordinary one-node-to-one-node graph edge.

    YAML example:

        - from: planner
          to: statistician
    """

    from_node: str = Field(
        alias="from"
    )

    to_node: str = Field(
        alias="to"
    )


# ============================================================
# Parallel group
# ============================================================

class WorkflowParallelGroup(
    BaseModel
):
    """
    Declarative fan-out / fan-in structure.

    Example YAML:

        parallel_groups:

          - from: planner

            branches:
              - researcher
              - statistician

            join: critic


    Meaning:

                       researcher
                      /          \\
        planner -----              ---- critic
                      \\          /
                       statistician


    The join node must wait for every branch in the group.
    """

    # --------------------------------------------------------
    # Node that creates the fan-out
    # --------------------------------------------------------

    from_node: str = Field(
        alias="from"
    )


    # --------------------------------------------------------
    # Independent downstream branches
    # --------------------------------------------------------

    branches: list[
        str
    ]


    # --------------------------------------------------------
    # Node that waits for all branches
    # --------------------------------------------------------

    join: str


# ============================================================
# Complete workflow definition
# ============================================================

class WorkflowDefinition(
    BaseModel
):
    """
    Complete declarative workflow specification.

    This is becoming our small workflow DSL.
    """

    name: str

    version: str


    # --------------------------------------------------------
    # Workflow inputs
    # --------------------------------------------------------

    inputs: dict[
        str,
        str,
    ]


    # --------------------------------------------------------
    # Global grounding rules
    # --------------------------------------------------------

    grounding_rules: list[
        str
    ]


    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    nodes: dict[
        str,
        WorkflowNode,
    ]


    # --------------------------------------------------------
    # Ordinary sequential edges
    # --------------------------------------------------------

    edges: list[
        WorkflowEdge
    ] = Field(
        default_factory=list
    )


    # --------------------------------------------------------
    # Step 10:
    # parallel fan-out / fan-in groups
    # --------------------------------------------------------

    parallel_groups: list[
        WorkflowParallelGroup
    ] = Field(
        default_factory=list
    )


    # --------------------------------------------------------
    # Start node
    # --------------------------------------------------------

    start: str


    # --------------------------------------------------------
    # Logical end node
    #
    # A conditional node may route directly to __end__,
    # as we implemented in Step 9.
    # --------------------------------------------------------

    end: str