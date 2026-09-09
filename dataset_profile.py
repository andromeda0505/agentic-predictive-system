from pathlib import Path
from typing import Any

import pandas as pd


def build_dataset_profile(
    path: Path,
) -> dict[str, Any]:

    df = pd.read_csv(path)

    profile: dict[str, Any] = {}

    # ========================================================
    # Basic dataset facts
    # ========================================================

    profile["row_count"] = int(len(df))

    profile["column_count"] = int(
        len(df.columns)
    )

    profile["column_names"] = (
        df.columns.tolist()
    )


    # ========================================================
    # Column profiles
    # ========================================================

    column_profiles = {}

    for column in df.columns:

        s = df[column]

        missing_count = int(
            s.isna().sum()
        )

        unique_count = int(
            s.nunique(dropna=True)
        )

        examples = (
            s
            .dropna()
            .astype(str)
            .head(5)
            .tolist()
        )

        column_profiles[column] = {

            "dtype": str(s.dtype),

            "missing_count":
                missing_count,

            "missing_percent":
                round(
                    100
                    * missing_count
                    / len(df),
                    3,
                ),

            "unique_count":
                unique_count,

            "example_values":
                examples,
        }

    profile["columns"] = (
        column_profiles
    )


    # ========================================================
    # Structural checks
    # ========================================================

    checks = {}

    checks[
        "has_construction_start_date"
    ] = (
        "construction_start_date"
        in df.columns
    )

    checks[
        "has_construction_end_date"
    ] = (
        "construction_end_date"
        in df.columns
    )

    checks[
        "has_physical_target_column"
    ] = (
        "construction_cycle_days"
        in df.columns
    )

    checks[
        "has_home_id"
    ] = (
        "home_id"
        in df.columns
    )


    # ========================================================
    # Repeated IDs versus duplicate rows
    # ========================================================

    if "home_id" in df.columns:

        id_counts = (
            df["home_id"]
            .value_counts(dropna=False)
        )

        unique_ids = int(
            df["home_id"]
            .nunique(dropna=True)
        )

        repeated_id_groups = int(
            (id_counts > 1).sum()
        )

        rows_in_repeated_groups = int(
            id_counts[
                id_counts > 1
            ].sum()
        )

        exact_duplicate_rows = int(
            df.duplicated().sum()
        )

        checks[
            "unique_home_ids"
        ] = unique_ids

        checks[
            "row_minus_unique_home_ids"
        ] = (
            len(df) - unique_ids
        )

        checks[
            "repeated_home_id_exists"
        ] = bool(
            repeated_id_groups > 0
        )

        checks[
            "repeated_home_id_group_count"
        ] = repeated_id_groups

        checks[
            "rows_in_repeated_home_id_groups"
        ] = rows_in_repeated_groups

        checks[
            "maximum_rows_per_home_id"
        ] = int(
            id_counts.max()
        )

        checks[
            "exact_duplicate_row_count"
        ] = exact_duplicate_rows


    # ========================================================
    # Date parsing
    # ========================================================

    date_columns = [

        "sale_date",

        "permit_application_date",

        "permit_issue_date",

        "construction_start_date",

        "construction_end_date",

        "final_inspection_date",
    ]


    parsed_dates = {}

    for column in date_columns:

        if column not in df.columns:
            continue

        raw = df[column]

        parsed = pd.to_datetime(
            raw,
            errors="coerce",
        )

        parsed_dates[column] = parsed

        nonmissing_raw = int(
            raw.notna().sum()
        )

        parseable = int(
            parsed.notna().sum()
        )

        unparseable_nonmissing = (
            nonmissing_raw - parseable
        )

        checks[
            f"{column}_nonmissing_count"
        ] = nonmissing_raw

        checks[
            f"{column}_parseable_count"
        ] = parseable

        checks[
            f"{column}_unparseable_nonmissing_count"
        ] = int(
            unparseable_nonmissing
        )

        if parsed.notna().any():

            checks[
                f"{column}_min"
            ] = str(
                parsed.min().date()
            )

            checks[
                f"{column}_max"
            ] = str(
                parsed.max().date()
            )


    # ========================================================
    # Permit issue versus construction start
    # ========================================================

    if (
        "permit_issue_date"
        in parsed_dates
        and
        "construction_start_date"
        in parsed_dates
    ):

        permit = parsed_dates[
            "permit_issue_date"
        ]

        start = parsed_dates[
            "construction_start_date"
        ]

        valid = (
            permit.notna()
            &
            start.notna()
        )

        if valid.any():

            after_start = (
                permit[valid]
                >
                start[valid]
            )

            checks[
                "permit_issue_after_start_count"
            ] = int(
                after_start.sum()
            )

            checks[
                "permit_issue_after_start_percent"
            ] = round(
                float(
                    after_start.mean()
                    * 100
                ),
                3,
            )


    # ========================================================
    # Construction target derivation
    # ========================================================

    target_spec = {

        "name":
            "construction_cycle_days",

        "problem_type":
            "regression",

        "prediction_time":
            "construction_start",

        "physical_column_present":
            (
                "construction_cycle_days"
                in df.columns
            ),

        "derivation":
            (
                "construction_end_date "
                "- construction_start_date"
            ),

        "derivable_from_existing_columns":
            False,
    }


    if (
        "construction_start_date"
        in parsed_dates
        and
        "construction_end_date"
        in parsed_dates
    ):

        target_spec[
            "derivable_from_existing_columns"
        ] = True

        start = parsed_dates[
            "construction_start_date"
        ]

        end = parsed_dates[
            "construction_end_date"
        ]

        both_dates = (
            start.notna()
            &
            end.notna()
        )

        raw_duration = (
            end - start
        ).dt.days

        invalid_negative = (
            both_dates
            &
            (raw_duration < 0)
        )

        valid_target = (
            both_dates
            &
            (raw_duration >= 0)
        )

        target_spec[
            "rows_with_both_target_dates"
        ] = int(
            both_dates.sum()
        )

        target_spec[
            "invalid_negative_duration_count"
        ] = int(
            invalid_negative.sum()
        )

        target_spec[
            "valid_derived_target_count"
        ] = int(
            valid_target.sum()
        )

        target_spec[
            "rows_without_usable_target"
        ] = int(
            len(df)
            -
            valid_target.sum()
        )

        if valid_target.any():

            valid_duration = (
                raw_duration[
                    valid_target
                ]
            )

            target_spec[
                "derived_target_min"
            ] = int(
                valid_duration.min()
            )

            target_spec[
                "derived_target_max"
            ] = int(
                valid_duration.max()
            )

            target_spec[
                "derived_target_mean"
            ] = round(
                float(
                    valid_duration.mean()
                ),
                3,
            )

            target_spec[
                "derived_target_median"
            ] = float(
                valid_duration.median()
            )


    profile[
        "target_specification"
    ] = target_spec


    # ========================================================
    # Construction status facts
    # ========================================================

    if (
        "construction_status"
        in df.columns
    ):

        status_counts = (
            df[
                "construction_status"
            ]
            .fillna("<MISSING>")
            .astype(str)
            .value_counts()
            .to_dict()
        )

        profile[
            "construction_status_counts"
        ] = {
            str(k): int(v)
            for k, v
            in status_counts.items()
        }


    # ========================================================
    # End-date missingness by construction status
    # ========================================================

    if (
        "construction_status"
        in df.columns
        and
        "construction_end_date"
        in df.columns
    ):

        temp = pd.DataFrame({

            "status":
                df["construction_status"]
                .fillna("<MISSING>")
                .astype(str),

            "end_missing":
                df[
                    "construction_end_date"
                ].isna(),
        })

        grouped = (
            temp
            .groupby("status")
            ["end_missing"]
            .agg(
                ["count", "sum"]
            )
        )

        profile[
            "end_date_missingness_by_status"
        ] = {

            str(index): {

                "rows":
                    int(row["count"]),

                "missing_end_dates":
                    int(row["sum"]),

            }

            for index, row
            in grouped.iterrows()
        }


    profile[
        "deterministic_checks"
    ] = checks

    return profile