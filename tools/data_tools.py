from pathlib import Path
from typing import Any, Callable
import json

import pandas as pd


def build_data_tools(
    state: dict[str, Any],
) -> dict[str, Callable]:

    """
    Build read-only dataset tools bound to the
    current workflow state.

    These tools inspect and analyze the dataset.
    They do not modify the original CSV.
    """

    # ========================================================
    # Load current workflow data
    # ========================================================

    dataset_path = Path(
        state["dataset_path"]
    )

    df = pd.read_csv(
        dataset_path
    )

    dataset_facts = state[
        "dataset_facts"
    ]

    feature_policy = state[
        "feature_policy"
    ]


    # ========================================================
    # Helper: Feature-policy lookup
    # ========================================================

    def policy_rule(
        feature: str,
    ) -> dict[str, Any]:
        """
        Return the feature-policy rule for one feature.
        """

        if feature not in feature_policy:

            return {
                "decision": "unknown",
                "reason":
                    "Feature is not present in "
                    "the feature policy.",
            }

        rule = feature_policy[
            feature
        ]

        # If rule happens to be a Pydantic object
        if hasattr(
            rule,
            "model_dump",
        ):

            return rule.model_dump()

        # Otherwise it should already be dictionary-like
        return dict(rule)


    # ========================================================
    # Helper: Build valid historical target sample
    # ========================================================

    def valid_target_frame():
        """
        Construct the historical regression sample.

        Target:
            construction_cycle_days
            =
            construction_end_date
            -
            construction_start_date

        Only rows with:
        - parseable start date;
        - parseable end date;
        - non-negative duration

        are retained.
        """

        temp = df.copy()

        start = pd.to_datetime(
            temp[
                "construction_start_date"
            ],
            errors="coerce",
        )

        end = pd.to_datetime(
            temp[
                "construction_end_date"
            ],
            errors="coerce",
        )

        duration = (
            end - start
        ).dt.days

        valid = (
            start.notna()
            &
            end.notna()
            &
            (duration >= 0)
        )

        result = (
            temp.loc[
                valid
            ]
            .copy()
        )

        result[
            "construction_cycle_days"
        ] = duration.loc[
            valid
        ]

        return result


    target_df = (
        valid_target_frame()
    )


    # ========================================================
    # TOOL 1 — Dataset Overview
    # ========================================================

    def dataset_overview() -> str:
        """
        Return authoritative high-level facts about the dataset.

        Returns:
            JSON containing:
            - dataset dimensions;
            - target specification;
            - construction-status counts;
            - duplicate information.
        """

        checks = dataset_facts[
            "deterministic_checks"
        ]

        result = {

            "row_count":
                dataset_facts[
                    "row_count"
                ],

            "column_count":
                dataset_facts[
                    "column_count"
                ],

            "target_specification":
                dataset_facts[
                    "target_specification"
                ],

            "construction_status_counts":
                dataset_facts[
                    "construction_status_counts"
                ],

            "exact_duplicate_row_count":
                checks.get(
                    "exact_duplicate_row_count"
                ),

            "repeated_home_id_group_count":
                checks.get(
                    "repeated_home_id_group_count"
                ),
        }

        return json.dumps(
            result,
            indent=2,
            default=str,
        )


    # ========================================================
    # TOOL 2 — Target Summary
    # ========================================================

    def target_summary() -> str:
        """
        Calculate the observed distribution of the valid
        derived target construction_cycle_days.

        Returns:
            JSON containing target sample size,
            quantiles, mean, median, range,
            and standard deviation.
        """

        y = target_df[
            "construction_cycle_days"
        ]

        result = {

            "valid_target_rows":
                int(
                    len(y)
                ),

            "minimum_days":
                float(
                    y.min()
                ),

            "q10":
                float(
                    y.quantile(
                        0.10
                    )
                ),

            "q25":
                float(
                    y.quantile(
                        0.25
                    )
                ),

            "median":
                float(
                    y.median()
                ),

            "mean":
                float(
                    y.mean()
                ),

            "q75":
                float(
                    y.quantile(
                        0.75
                    )
                ),

            "q90":
                float(
                    y.quantile(
                        0.90
                    )
                ),

            "maximum_days":
                float(
                    y.max()
                ),

            "standard_deviation":
                float(
                    y.std()
                ),
        }

        return json.dumps(
            result,
            indent=2,
        )


    # ========================================================
    # TOOL 3 — Duplicate Summary
    # ========================================================

    def duplicate_summary() -> str:
        """
        Inspect repeated home IDs and exact duplicate rows.

        This deliberately distinguishes:

        repeated identifiers
            versus

        completely identical rows.

        Returns:
            JSON containing duplicate statistics.
        """

        id_counts = (
            df[
                "home_id"
            ]
            .value_counts(
                dropna=False
            )
        )

        repeated = (
            id_counts[
                id_counts > 1
            ]
        )

        result = {

            "total_rows":
                int(
                    len(df)
                ),

            "unique_home_ids":
                int(
                    df[
                        "home_id"
                    ]
                    .nunique(
                        dropna=True
                    )
                ),

            "repeated_home_id_groups":
                int(
                    len(repeated)
                ),

            "rows_in_repeated_groups":
                int(
                    repeated.sum()
                ),

            "maximum_rows_per_home_id":
                int(
                    id_counts.max()
                ),

            "exact_duplicate_rows":
                int(
                    df
                    .duplicated()
                    .sum()
                ),
        }

        return json.dumps(
            result,
            indent=2,
        )


    # ========================================================
    # TOOL 4 — Candidate Predictor Policy
    # ========================================================

    def list_candidate_predictors() -> str:
        """
        List features according to the prediction-time
        feature contract.

        Returns:
            JSON grouping features into:
            - allow
            - verify
            - block
            - identifier
            - reference_time
            - training_only
        """

        grouped = {

            "allow": [],

            "verify": [],

            "block": [],

            "identifier": [],

            "reference_time": [],

            "training_only": [],
        }

        for (
            feature,
            rule,
        ) in feature_policy.items():

            if hasattr(
                rule,
                "model_dump",
            ):

                rule = (
                    rule.model_dump()
                )

            decision = rule[
                "decision"
            ]

            if decision not in grouped:

                grouped[
                    decision
                ] = []

            grouped[
                decision
            ].append(
                feature
            )

        # Sort for stable, readable output
        for decision in grouped:

            grouped[
                decision
            ] = sorted(
                grouped[
                    decision
                ]
            )

        return json.dumps(
            grouped,
            indent=2,
        )


    # ========================================================
    # TOOL 5 — Allowed Predictor Missingness
    # ========================================================

    def allowed_predictor_missingness() -> str:
        """
        Calculate missingness for every predictor whose
        feature-policy decision is 'allow'.

        Returns:
            JSON containing:
            - missing count;
            - missing percentage;
            - non-missing count

            for every allowed predictor.
        """

        result = {}


        for (
            feature,
            rule,
        ) in feature_policy.items():

            if hasattr(
                rule,
                "model_dump",
            ):

                rule = (
                    rule.model_dump()
                )


            if (
                rule[
                    "decision"
                ]
                !=
                "allow"
            ):

                continue


            # Defensive check
            if feature not in df.columns:

                result[
                    feature
                ] = {
                    "error":
                        "Feature appears in the "
                        "feature policy but not "
                        "in the dataset."
                }

                continue


            s = df[
                feature
            ]

            missing_count = int(
                s
                .isna()
                .sum()
            )

            result[
                feature
            ] = {

                "missing_count":
                    missing_count,

                "missing_percent":
                    round(
                        100
                        *
                        s
                        .isna()
                        .mean(),
                        3,
                    ),

                "nonmissing_count":
                    int(
                        len(s)
                        -
                        missing_count
                    ),
            }


        return json.dumps(
            result,
            indent=2,
        )


    # ========================================================
    # TOOL 6 — Inspect One Feature
    # ========================================================

    def inspect_feature(
        feature: str,
    ) -> str:
        """
        Inspect one dataset feature and its prediction-time
        feature-policy rule.

        Args:
            feature:
                Exact dataset column name.

        Returns:
            JSON containing:
            - dtype;
            - missingness;
            - cardinality;
            - examples;
            - prediction-time policy.
        """

        if feature not in df.columns:

            return json.dumps({

                "error":
                    f"Column '{feature}' "
                    "does not exist.",

                "valid_columns":
                    list(
                        df.columns
                    ),
            })


        s = df[
            feature
        ]

        result = {

            "feature":
                feature,

            "dtype":
                str(
                    s.dtype
                ),

            "missing_count":
                int(
                    s
                    .isna()
                    .sum()
                ),

            "missing_percent":
                round(
                    100
                    *
                    s
                    .isna()
                    .mean(),
                    3,
                ),

            "unique_count":
                int(
                    s
                    .nunique(
                        dropna=True
                    )
                ),

            "example_values":
                (
                    s
                    .dropna()
                    .astype(str)
                    .head(10)
                    .tolist()
                ),

            "feature_policy":
                policy_rule(
                    feature
                ),
        }

        return json.dumps(
            result,
            indent=2,
            default=str,
        )


    # ========================================================
    # TOOL 7 — Numeric Candidate Analysis
    # ========================================================

    def analyze_numeric_candidate(
        feature: str,
    ) -> str:
        """
        Analyze one allowed numeric predictor against
        construction_cycle_days.

        Args:
            feature:
                Exact dataset column name.

        Returns:
            JSON containing:
            - valid sample size;
            - predictor mean/std;
            - Pearson correlation;
            - Spearman rank correlation.

        Important:
            This tool reports association only.
            It does not imply causality.
        """

        if feature not in df.columns:

            return json.dumps({

                "error":
                    f"Column '{feature}' "
                    "does not exist."
            })


        rule = policy_rule(
            feature
        )


        if (
            rule[
                "decision"
            ]
            !=
            "allow"
        ):

            return json.dumps({

                "error":
                    "This tool only analyzes "
                    "features whose prediction-time "
                    "policy is 'allow'.",

                "feature":
                    feature,

                "policy":
                    rule,
            })


        x = pd.to_numeric(
            target_df[
                feature
            ],
            errors="coerce",
        )

        y = target_df[
            "construction_cycle_days"
        ]


        valid = (
            x.notna()
            &
            y.notna()
        )


        if int(
            valid.sum()
        ) < 3:

            return json.dumps({

                "error":
                    "Insufficient valid numeric "
                    "observations.",

                "feature":
                    feature,

                "valid_numeric_rows":
                    int(
                        valid.sum()
                    ),
            })


        x_valid = (
            x[
                valid
            ]
        )

        y_valid = (
            y[
                valid
            ]
        )


        pearson = (
            x_valid
            .corr(
                y_valid,
                method="pearson",
            )
        )


        spearman = (
            x_valid
            .corr(
                y_valid,
                method="spearman",
            )
        )


        result = {

            "feature":
                feature,

            "policy":
                rule,

            "sample_size":
                int(
                    valid.sum()
                ),

            "missing_or_non_numeric_rows":
                int(
                    len(target_df)
                    -
                    valid.sum()
                ),

            "predictor_mean":
                float(
                    x_valid.mean()
                ),

            "predictor_std":
                float(
                    x_valid.std()
                ),

            "pearson_correlation_with_target":
                (
                    None

                    if pd.isna(
                        pearson
                    )

                    else float(
                        pearson
                    )
                ),

            "spearman_correlation_with_target":
                (
                    None

                    if pd.isna(
                        spearman
                    )

                    else float(
                        spearman
                    )
                ),

            "interpretation_guardrail":
                (
                    "Observed correlation is an "
                    "association and does not establish "
                    "causal effect."
                ),
        }


        return json.dumps(
            result,
            indent=2,
        )


    # ========================================================
    # TOOL 8 — Categorical Candidate Analysis
    # ========================================================

    def analyze_categorical_candidate(
        feature: str,
    ) -> str:
        """
        Compare construction-cycle duration across categories
        of one allowed predictor.

        Args:
            feature:
                Exact categorical dataset column name.

        Returns:
            JSON containing group counts,
            target means, and target medians
            for the 20 largest groups.
        """

        if feature not in df.columns:

            return json.dumps({

                "error":
                    f"Column '{feature}' "
                    "does not exist."
            })


        rule = policy_rule(
            feature
        )


        if (
            rule[
                "decision"
            ]
            !=
            "allow"
        ):

            return json.dumps({

                "error":
                    "This tool only analyzes "
                    "features whose prediction-time "
                    "policy is 'allow'.",

                "feature":
                    feature,

                "policy":
                    rule,
            })


        temp = target_df[
            [
                feature,
                "construction_cycle_days",
            ]
        ].copy()


        temp[
            feature
        ] = (
            temp[
                feature
            ]
            .fillna(
                "<MISSING>"
            )
            .astype(str)
        )


        grouped = (

            temp
            .groupby(
                feature
            )
            [
                "construction_cycle_days"
            ]
            .agg(
                count="count",
                mean="mean",
                median="median",
            )
            .sort_values(
                "count",
                ascending=False,
            )
            .head(20)
            .reset_index()
        )


        records = (
            grouped
            .to_dict(
                orient="records"
            )
        )


        # Convert NumPy/Pandas numeric objects
        # into ordinary Python values.

        clean_records = []

        for record in records:

            clean_records.append({

                feature:
                    str(
                        record[
                            feature
                        ]
                    ),

                "count":
                    int(
                        record[
                            "count"
                        ]
                    ),

                "mean_target_days":
                    float(
                        record[
                            "mean"
                        ]
                    ),

                "median_target_days":
                    float(
                        record[
                            "median"
                        ]
                    ),
            })


        result = {

            "feature":
                feature,

            "policy":
                rule,

            "group_count":
                int(
                    temp[
                        feature
                    ]
                    .nunique()
                ),

            "largest_groups":
                clean_records,

            "interpretation_guardrail":
                (
                    "Differences between category "
                    "means are descriptive associations "
                    "and do not establish causal effects."
                ),
        }


        return json.dumps(
            result,
            indent=2,
            default=str,
        )


    # ========================================================
    # TOOL 9 — Temporal Validation Feasibility
    # ========================================================

    def check_time_split_feasibility() -> str:
        """
        Inspect whether temporal validation is feasible.

        Uses only rows with a valid historical target.

        Returns:
            JSON containing:
            - usable date range;
            - target observations per construction-start year;
            - whether multiple years are available.

        Important:
            This tool determines feasibility only.
            It does not automatically choose the optimal cutoff.
        """

        start = pd.to_datetime(

            target_df[
                "construction_start_date"
            ],

            errors="coerce",
        )


        valid = (
            start.notna()
        )


        if int(
            valid.sum()
        ) == 0:

            return json.dumps({

                "valid_target_rows_with_start_date":
                    0,

                "temporal_holdout_feasible":
                    False,

                "error":
                    "No valid construction-start "
                    "dates are available in the "
                    "historical target sample."
            })


        years = (

            start[
                valid
            ]
            .dt.year
            .value_counts()
            .sort_index()
        )


        result = {

            "valid_target_rows_with_start_date":
                int(
                    valid.sum()
                ),

            "minimum_start_date":
                str(
                    start[
                        valid
                    ]
                    .min()
                    .date()
                ),

            "maximum_start_date":
                str(
                    start[
                        valid
                    ]
                    .max()
                    .date()
                ),

            "rows_by_start_year": {

                str(
                    int(year)
                ):
                    int(count)

                for (
                    year,
                    count
                )
                in years.items()
            },

            "distinct_start_years":
                int(
                    len(years)
                ),

            "temporal_holdout_feasible":
                bool(
                    len(years) >= 2
                ),

            "interpretation_guardrail":
                (
                    "Temporal holdout feasibility does "
                    "not itself determine the optimal "
                    "training/validation cutoff."
                ),
        }


        return json.dumps(
            result,
            indent=2,
        )


    # ========================================================
    # TOOL 10 — Censoring Summary
    # ========================================================

    def censoring_summary() -> str:
        """
        Inspect unfinished observations and possible
        right censoring.

        Returns:
            JSON containing:
            - construction-status counts;
            - end-date missingness by status;
            - an interpretation guardrail.
        """

        result = {

            "construction_status_counts":
                dataset_facts[
                    "construction_status_counts"
                ],

            "end_date_missingness_by_status":
                dataset_facts[
                    "end_date_missingness_by_status"
                ],

            "interpretation_guardrail":
                (
                    "In-progress observations without "
                    "construction_end_date should not "
                    "automatically be treated as ordinary "
                    "randomly missing regression labels. "
                    "They may represent right-censored "
                    "duration observations."
                ),
        }


        return json.dumps(
            result,
            indent=2,
        )


    # ========================================================
    # Registry returned to caller
    # ========================================================

    return {

        "dataset_overview":
            dataset_overview,

        "target_summary":
            target_summary,

        "duplicate_summary":
            duplicate_summary,

        "list_candidate_predictors":
            list_candidate_predictors,

        "allowed_predictor_missingness":
            allowed_predictor_missingness,

        "inspect_feature":
            inspect_feature,

        "analyze_numeric_candidate":
            analyze_numeric_candidate,

        "analyze_categorical_candidate":
            analyze_categorical_candidate,

        "check_time_split_feasibility":
            check_time_split_feasibility,

        "censoring_summary":
            censoring_summary,
    }