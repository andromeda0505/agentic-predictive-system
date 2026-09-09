from pydantic import BaseModel

from schemas.agent_outputs import (
    PlannerOutput,
    StatisticianOutput,
    CriticOutput,
)


OUTPUT_SCHEMAS = {

    "planner_v1":
        PlannerOutput,

    "statistician_v1":
        StatisticianOutput,

    "critic_v1":
        CriticOutput,
}


def get_output_model(
    schema_name: str,
) -> type[BaseModel]:

    if (
        schema_name
        not in OUTPUT_SCHEMAS
    ):

        raise ValueError(
            "Unknown output schema: "
            f"{schema_name}"
        )

    return OUTPUT_SCHEMAS[
        schema_name
    ]
