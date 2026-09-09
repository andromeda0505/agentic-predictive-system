from __future__ import annotations

from typing import Any

import json
import re


# ============================================================
# Generic helpers
# ============================================================

def _rule_as_dict(
    rule,
) -> dict[str, Any]:

    if hasattr(
        rule,
        "model_dump",
    ):
        return rule.model_dump()

    return dict(
        rule
    )


def _get_structured_artifact(
    state: dict[str, Any],
    artifact_name: str,
) -> dict[str, Any]:

    artifacts = state.get(
        "artifacts",
        {},
    )

    artifact = artifacts.get(
        artifact_name,
        {},
    )

    if hasattr(
        artifact,
        "model_dump",
    ):
        artifact = artifact.model_dump()

    structured_data = artifact.get(
        "structured_data"
    )

    if structured_data is None:
        return {}

    return structured_data


def _state_with_candidate_artifact(
    state: dict[str, Any],
    artifact_name: str,
    structured_data: dict[str, Any],
) -> dict[str, Any]:

    candidate_state = dict(
        state
    )

    existing_artifacts = dict(
        state.get(
            "artifacts",
            {},
        )
    )

    existing_artifacts[
        artifact_name
    ] = {
        "structured_data":
            structured_data
    }

    candidate_state[
        "artifacts"
    ] = existing_artifacts

    return candidate_state


def _collect_strings(
    value: Any,
) -> list[str]:

    result: list[str] = []

    if isinstance(
        value,
        str,
    ):

        result.append(
            value
        )

    elif isinstance(
        value,
        dict,
    ):

        for item in value.values():

            result.extend(
                _collect_strings(
                    item
                )
            )

    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        for item in value:

            result.extend(
                _collect_strings(
                    item
                )
            )

    return result


def _normalized(
    value: str,
) -> str:

    return (
        value
        .strip()
        .lower()
        .replace(
            "-",
            " ",
        )
        .replace(
            "_",
            " ",
        )
    )


# ============================================================
# Censoring helpers
# ============================================================

def _deterministic_censoring_present(
    state: dict[str, Any],
) -> tuple[
    bool,
    int,
]:
    """
    Determine from authoritative state whether unfinished
    homes with missing construction_end_date are present.
    """

    dataset_facts = state.get(
        "dataset_facts",
        {},
    )

    status_counts = dataset_facts.get(
        "construction_status_counts",
        {},
    )

    missingness = dataset_facts.get(
        "end_date_missingness_by_status",
        {},
    )

    in_progress = int(
        status_counts.get(
            "In Progress",
            0,
        )
    )

    missing_end_dates = int(
        missingness
        .get(
            "In Progress",
            {},
        )
        .get(
            "missing_end_dates",
            0,
        )
    )

    present = (
        in_progress > 0
        and
        missing_end_dates > 0
    )

    return (
        present,
        missing_end_dates,
    )


def _is_positive_end_date_imputation_claim(
    value: str,
) -> bool:
    """
    True only when a statement positively recommends or asks
    about imputing construction_end_date.

    Correct statements such as:

        "do not impute construction_end_date"

    are intentionally allowed.
    """

    lower = (
        str(
            value
        )
        .strip()
        .lower()
    )

    if (
        "construction_end_date"
        not in lower
    ):
        return False


    allowed_negations = [
        "do not impute",
        "don't impute",
        "must not impute",
        "should not impute",
        "should not be imputed",
        "must not be imputed",
        "avoid imputing",
        "avoid imputation",
        "not ordinary imputation",
        "not use ordinary imputation",
        "not recommend imputation",
        "imputation is inappropriate",
        "imputation is not appropriate",
    ]


    if any(
        phrase in lower

        for phrase
        in allowed_negations
    ):

        return False


    return (
        re.search(
            r"\bimput",
            lower,
        )
        is not None
    )


# ============================================================
# NEW STEP 10E:
# deterministic repair for one hard semantic contract
# ============================================================

def _repair_statistician_censoring_contract(
    parsed_output,
    state: dict[str, Any],
) -> bool:
    """
    Repair ONLY the narrow hard-contract violation:

        construction_end_date
                +
        ordinary imputation recommendation

    when deterministic evidence establishes potential
    right-censoring.

    WHY REPAIR INSTEAD OF RETRY FOREVER?

    This is not an open statistical judgment. The system has
    already established that construction_end_date creates the
    target and is missing for unfinished observations.

    Therefore recommending ordinary imputation of that field
    violates a deterministic contract.

    Other semantic mistakes are NOT repaired here. They still
    fail validation normally.
    """

    (
        censoring_present,
        missing_end_dates,
    ) = _deterministic_censoring_present(
        state
    )


    if not censoring_present:

        return False


    repaired = False


    # ========================================================
    # 1. sample_handling
    # ========================================================

    sample_handling = list(
        getattr(
            parsed_output,
            "sample_handling",
            [],
        )
    )


    new_sample_handling: list[str] = []


    for item in sample_handling:

        if _is_positive_end_date_imputation_claim(
            item
        ):

            new_sample_handling.append(
                (
                    "Treat unfinished homes with missing "
                    "construction_end_date as potential "
                    "right-censored outcomes; compare "
                    "completed-case and censoring-aware "
                    "approaches."
                )
            )

            repaired = True

        else:

            new_sample_handling.append(
                item
            )


    if repaired:

        parsed_output.sample_handling = list(
            dict.fromkeys(
                new_sample_handling
            )
        )


    # ========================================================
    # 2. missing_data_strategy
    # ========================================================

    missing_strategy = list(
        getattr(
            parsed_output,
            "missing_data_strategy",
            [],
        )
    )


    new_missing_strategy: list[str] = []


    local_repair = False


    for item in missing_strategy:

        if _is_positive_end_date_imputation_claim(
            item
        ):

            new_missing_strategy.append(
                (
                    "Apply model-compatible imputation only "
                    "to prediction-time predictors with "
                    "ordinary missingness; handle missing "
                    "construction_end_date separately as "
                    "potential outcome censoring."
                )
            )

            repaired = True
            local_repair = True

        else:

            new_missing_strategy.append(
                item
            )


    if local_repair:

        parsed_output.missing_data_strategy = list(
            dict.fromkeys(
                new_missing_strategy
            )
        )


    # ========================================================
    # 3. unresolved_questions
    #
    # This is where Qwen is currently stuck.
    # ========================================================

    unresolved = list(
        getattr(
            parsed_output,
            "unresolved_questions",
            [],
        )
    )


    new_unresolved: list[str] = []

    local_repair = False


    for item in unresolved:

        if _is_positive_end_date_imputation_claim(
            item
        ):

            new_unresolved.append(
                (
                    "How sensitive are conclusions to "
                    "alternative censoring-aware handling "
                    "of unfinished observations?"
                )
            )

            repaired = True
            local_repair = True

        else:

            new_unresolved.append(
                item
            )


    if local_repair:

        parsed_output.unresolved_questions = list(
            dict.fromkeys(
                new_unresolved
            )
        )


    # ========================================================
    # 4. handoff
    # ========================================================

    handoff = str(
        getattr(
            parsed_output,
            "handoff",
            "",
        )
    )


    if _is_positive_end_date_imputation_claim(
        handoff
    ):

        parsed_output.handoff = (
            "Proceed by quantifying ordinary missingness "
            "among prediction-time predictors, using "
            "model-compatible predictor imputation when "
            "appropriate, implementing temporal validation, "
            "evaluating regression candidates against "
            "baselines, and comparing completed-case "
            "regression with a censoring-aware survival "
            "analysis candidate for unfinished homes."
        )

        repaired = True


    # ========================================================
    # 5. censoring.note
    # ========================================================

    censoring = getattr(
        parsed_output,
        "censoring",
        None,
    )


    if (
        censoring is not None
    ):

        note = str(
            getattr(
                censoring,
                "note",
                "",
            )
        )


        if _is_positive_end_date_imputation_claim(
            note
        ):

            censoring.note = (
                "Deterministic evidence shows "
                f"{missing_end_dates} unfinished "
                "observations with missing "
                "construction_end_date; treat these as "
                "potential right-censored outcomes rather "
                "than ordinary missing values to impute."
            )

            repaired = True


    return repaired


# ============================================================
# Planner audit
# ============================================================

def _deterministic_planner_audit(
    state: dict[str, Any],
) -> list[dict[str, str]]:

    planner = _get_structured_artifact(
        state,
        "planner",
    )


    target_spec = (
        state[
            "dataset_facts"
        ][
            "target_specification"
        ]
    )


    target_name = str(
        target_spec[
            "name"
        ]
    )


    problem_type = (
        str(
            target_spec[
                "problem_type"
            ]
        )
        .lower()
    )


    findings: list[
        dict[str, str]
    ] = []


    planner_text = (
        "\n"
        .join(
            _collect_strings(
                planner
            )
        )
        .lower()
    )


    # ========================================================
    # Target
    # ========================================================

    target_description = (
        str(
            planner.get(
                "target_construction",
                "",
            )
        )
        .lower()
    )


    if (
        target_name.lower()
        not in target_description
    ):

        findings.append(
            {
                "source_agent":
                    "planner",

                "category":
                    "target_definition",

                "message":
                    (
                        "Planner target description does not "
                        "identify the authoritative target "
                        f"'{target_name}'."
                    ),
            }
        )


    # ========================================================
    # Regression vs classification
    # ========================================================

    if (
        problem_type
        ==
        "regression"
    ):

        classification_phrases = [
            "binary outcome",
            "binary classification",
            "classification task",
            "completed or not completed",
            "completion rate",
        ]


        if any(
            phrase in planner_text

            for phrase
            in classification_phrases
        ):

            findings.append(
                {
                    "source_agent":
                        "planner",

                    "category":
                        "target_definition",

                    "message":
                        (
                            "Planner converts the "
                            "authoritative regression problem "
                            "into classification."
                        ),
                }
            )


    # ========================================================
    # Model choices
    # ========================================================

    if (
        problem_type
        ==
        "regression"
    ):

        for model in planner.get(
            "candidate_models",
            [],
        ):

            model_name = _normalized(
                str(
                    model.get(
                        "name",
                        "",
                    )
                )
            )


            if (
                "logistic regression"
                in model_name
            ):

                findings.append(
                    {
                        "source_agent":
                            "planner",

                        "category":
                            "model_choice",

                        "message":
                            (
                                "Planner proposes Logistic "
                                "Regression for the continuous "
                                "regression target."
                            ),
                    }
                )


    # ========================================================
    # Metrics
    # ========================================================

    if (
        problem_type
        ==
        "regression"
    ):

        classification_metrics = {
            "accuracy",
            "f1",
            "f1 score",
            "precision",
            "recall",
            "auc",
            "roc auc",
        }


        for metric in planner.get(
            "metrics",
            [],
        ):

            metric_name = _normalized(
                str(
                    metric.get(
                        "name",
                        "",
                    )
                )
            )


            if (
                metric_name
                in classification_metrics
            ):

                findings.append(
                    {
                        "source_agent":
                            "planner",

                        "category":
                            "metric_choice",

                        "message":
                            (
                                "Planner proposes "
                                f"classification metric "
                                f"'{metric.get('name')}' "
                                "for regression."
                            ),
                    }
                )


    # ========================================================
    # Invented snake_case variables
    # ========================================================

    actual_columns = set(
        state[
            "dataset_facts"
        ][
            "column_names"
        ]
    )


    actual_columns.add(
        target_name
    )


    allowed_derived_names = {
        "construction_cycle_days",
        "construction_start_year",
        "construction_start_quarter",
        "construction_start_month",
        "construction_start_day_of_week",
        "construction_start_week",
        "construction_start_season",
        "day_of_week",
        "month",
        "quarter",
        "year",
        "week",
        "season",
    }


    fields_to_scan = [
        planner.get(
            "target_construction",
            "",
        ),
        planner.get(
            "data_quality_actions",
            [],
        ),
        planner.get(
            "feature_engineering",
            [],
        ),
        planner.get(
            "censoring_consideration",
            "",
        ),
        planner.get(
            "unresolved_questions",
            [],
        ),
    ]


    scan_text = "\n".join(
        _collect_strings(
            fields_to_scan
        )
    )


    tokens = set(
        re.findall(
            (
                r"\b[a-zA-Z][a-zA-Z0-9]*"
                r"(?:_[a-zA-Z0-9]+)+\b"
            ),
            scan_text,
        )
    )


    unknown_tokens = sorted(
        token
        for token in tokens
        if (
            token not in actual_columns
            and
            token not in allowed_derived_names
        )
    )


    if unknown_tokens:

        findings.append(
            {
                "source_agent":
                    "planner",

                "category":
                    "column_hallucination",

                "message":
                    (
                        "Planner references unknown "
                        "snake_case variables: "
                        f"{unknown_tokens}"
                    ),
            }
        )


    return findings


# ============================================================
# Statistician censoring audit
# ============================================================

def _statistician_censoring_imputation_audit(
    statistician: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, str]]:

    findings: list[
        dict[str, str]
    ] = []


    (
        censoring_present,
        missing_end_dates,
    ) = _deterministic_censoring_present(
        state
    )


    if not censoring_present:

        return findings


    fields_to_scan = [
        statistician.get(
            "sample_handling",
            [],
        ),
        statistician.get(
            "missing_data_strategy",
            [],
        ),
        statistician.get(
            "unresolved_questions",
            [],
        ),
        statistician.get(
            "handoff",
            "",
        ),
        statistician.get(
            "censoring",
            {},
        ),
    ]


    invalid_strings: list[str] = []


    for text in _collect_strings(
        fields_to_scan
    ):

        if _is_positive_end_date_imputation_claim(
            text
        ):

            invalid_strings.append(
                str(
                    text
                )
            )


    if invalid_strings:

        findings.append(
            {
                "source_agent":
                    "statistician",

                "category":
                    "censoring",

                "message":
                    (
                        "Statistician treats "
                        "construction_end_date as an ordinary "
                        "imputation target despite "
                        f"{missing_end_dates} unfinished "
                        "observations with missing end dates. "
                        "Predictor missingness and outcome "
                        "censoring must be handled separately."
                    ),
            }
        )


    return findings


# ============================================================
# Statistician audit
# ============================================================

def _deterministic_statistician_audit(
    state: dict[str, Any],
) -> list[dict[str, str]]:

    statistician = _get_structured_artifact(
        state,
        "statistician",
    )


    findings: list[
        dict[str, str]
    ] = []


    findings.extend(
        _statistician_censoring_imputation_audit(
            statistician,
            state,
        )
    )


    # ========================================================
    # Grounded feature set
    # ========================================================

    allowed_features: set[str] = set()

    reference_features: set[str] = set()


    for (
        feature,
        rule,
    ) in state[
        "feature_policy"
    ].items():

        rule_dict = _rule_as_dict(
            rule
        )


        decision = rule_dict[
            "decision"
        ]


        if (
            decision
            ==
            "allow"
        ):

            allowed_features.add(
                feature
            )


        elif (
            decision
            ==
            "reference_time"
        ):

            reference_features.add(
                feature
            )


    derived_calendar_features = {
        "construction_start_year",
        "construction_start_quarter",
        "construction_start_month",
        "construction_start_day_of_week",
        "construction_start_week",
        "construction_start_season",
        "year",
        "quarter",
        "month",
        "day_of_week",
        "week",
        "season",
    }


    grounded_features = (
        allowed_features
        |
        reference_features
        |
        derived_calendar_features
    )


    def explicitly_mentioned(
        feature: str,
        text: str,
    ) -> bool:

        pattern = (
            r"(?<![a-zA-Z0-9_])"
            +
            re.escape(
                feature.lower()
            )
            +
            r"(?![a-zA-Z0-9_])"
        )


        return (
            re.search(
                pattern,
                text,
            )
            is not None
        )


    def compact_interaction_pair(
        text: str,
    ) -> tuple[str, str] | None:

        text_lower = (
            text
            .strip()
            .lower()
        )


        snake_tokens = set(
            re.findall(
                (
                    r"\b[a-zA-Z][a-zA-Z0-9]*"
                    r"(?:_[a-zA-Z0-9]+)+\b"
                ),
                text_lower,
            )
        )


        snake_tokens.add(
            text_lower
        )


        features = sorted(
            grounded_features
        )


        for left in features:

            for right in features:

                if (
                    left
                    ==
                    right
                ):
                    continue


                valid_forms = {
                    f"{left}_{right}",
                    f"{left}_x_{right}",
                    f"{left}_by_{right}",
                }


                if any(
                    token in valid_forms

                    for token
                    in snake_tokens
                ):

                    return (
                        left,
                        right,
                    )


        return None


    # ========================================================
    # Interaction grounding
    # ========================================================

    for interaction in statistician.get(
        "interactions_to_investigate",
        [],
    ):

        interaction_text = (
            str(
                interaction
            )
            .strip()
            .lower()
        )


        explicit_features = {
            feature
            for feature in grounded_features
            if explicitly_mentioned(
                feature,
                interaction_text,
            )
        }


        if (
            len(
                explicit_features
            )
            >=
            2
        ):
            continue


        if (
            compact_interaction_pair(
                interaction_text
            )
            is not None
        ):
            continue


        findings.append(
            {
                "source_agent":
                    "statistician",

                "category":
                    "unsupported_claim",

                "message":
                    (
                        "Statistician proposes interaction "
                        f"'{interaction}' without at least "
                        "two grounded prediction-time "
                        "variables."
                    ),
            }
        )


    return findings


# ============================================================
# Combined audit
# ============================================================

def _deterministic_upstream_audit(
    state: dict[str, Any],
) -> list[dict[str, str]]:

    findings: list[
        dict[str, str]
    ] = []


    findings.extend(
        _deterministic_planner_audit(
            state
        )
    )


    findings.extend(
        _deterministic_statistician_audit(
            state
        )
    )


    return findings


# ============================================================
# Critic self-grounding
# ============================================================

def _critic_unknown_column_tokens(
    data: dict[str, Any],
    state: dict[str, Any],
) -> list[str]:

    actual_columns = set(
        state[
            "dataset_facts"
        ][
            "column_names"
        ]
    )


    target_name = (
        state[
            "dataset_facts"
        ][
            "target_specification"
        ][
            "name"
        ]
    )


    actual_columns.add(
        target_name
    )


    allowed_derived_names = {
        "construction_cycle_days",
        "construction_start_year",
        "construction_start_quarter",
        "construction_start_month",
        "construction_start_day_of_week",
        "construction_start_week",
        "construction_start_season",
        "day_of_week",
        "month",
        "quarter",
        "year",
        "week",
        "season",
    }


    recommended_fix_text = []


    for issue in data.get(
        "issues",
        [],
    ):

        recommended_fix_text.append(
            str(
                issue.get(
                    "recommended_fix",
                    "",
                )
            )
        )


    combined = "\n".join(
        recommended_fix_text
    )


    tokens = set(
        re.findall(
            (
                r"\b[a-zA-Z][a-zA-Z0-9]*"
                r"(?:_[a-zA-Z0-9]+)+\b"
            ),
            combined,
        )
    )


    return sorted(
        token
        for token in tokens
        if (
            token not in actual_columns
            and
            token not in allowed_derived_names
        )
    )


# ============================================================
# Public structured validator
# ============================================================

def validate_structured_output(
    node_name: str,
    parsed_output,
    state: dict[str, Any],
    node_spec,
):
    """
    Validate structured output before it enters shared state.

    Step 10E addition:

    A narrow deterministic repair is applied to the
    Statistician's construction_end_date/imputation contract
    BEFORE model_dump() and semantic auditing.

    Because parsed_output itself is mutated, the corrected
    object is also the artifact returned by structured_runtime.
    """


    # ========================================================
    # HARD-CONTRACT REPAIR FIRST
    # ========================================================

    if (
        node_name
        ==
        "statistician"
    ):

        repaired = (
            _repair_statistician_censoring_contract(
                parsed_output,
                state,
            )
        )


        if repaired:

            print()

            print(
                (
                    "DETERMINISTIC STATISTICIAN REPAIR "
                    "APPLIED: censoring/imputation contract"
                )
            )


    # ========================================================
    # Dump AFTER repair
    # ========================================================

    data = parsed_output.model_dump()


    errors: list[str] = []


    # ========================================================
    # PLANNER
    # ========================================================

    if (
        node_name
        ==
        "planner"
    ):

        candidate_state = (
            _state_with_candidate_artifact(
                state,
                "planner",
                data,
            )
        )


        findings = (
            _deterministic_planner_audit(
                candidate_state
            )
        )


        for finding in findings:

            errors.append(
                (
                    "Planner semantic validation failed "
                    f"[{finding['category']}]: "
                    f"{finding['message']}"
                )
            )


    # ========================================================
    # STATISTICIAN
    # ========================================================

    if (
        node_name
        ==
        "statistician"
    ):

        (
            censoring_expected,
            _,
        ) = _deterministic_censoring_present(
            state
        )


        observed_censoring = (
            data[
                "censoring"
            ][
                "right_censoring_present"
            ]
        )


        if (
            observed_censoring
            !=
            censoring_expected
        ):

            errors.append(
                (
                    "Statistician semantic validation failed "
                    "[censoring]: censoring assessment "
                    "contradicts deterministic evidence."
                )
            )


        # ----------------------------------------------------
        # No invented temporal cutoff
        # ----------------------------------------------------

        if (
            data[
                "validation"
            ][
                "cutoff_defined"
            ]
        ):

            errors.append(
                (
                    "Statistician semantic validation failed "
                    "[validation_design]: no deterministic "
                    "procedure selected an exact cutoff."
                )
            )


        candidate_state = (
            _state_with_candidate_artifact(
                state,
                "statistician",
                data,
            )
        )


        findings = (
            _deterministic_statistician_audit(
                candidate_state
            )
        )


        for finding in findings:

            errors.append(
                (
                    "Statistician semantic validation failed "
                    f"[{finding['category']}]: "
                    f"{finding['message']}"
                )
            )


    # ========================================================
    # CRITIC
    # ========================================================

    if (
        node_name
        ==
        "critic"
    ):

        verdict = data[
            "verdict"
        ]

        issues = data[
            "issues"
        ]

        recommended_action = data[
            "recommended_next_action"
        ]


        # ----------------------------------------------------
        # Internal consistency
        # ----------------------------------------------------

        if (
            verdict
            ==
            "pass"
        ):

            if issues:

                errors.append(
                    (
                        "Critic verdict is 'pass' "
                        "but issues were reported."
                    )
                )


            if (
                recommended_action
                !=
                "accept"
            ):

                errors.append(
                    (
                        "Critic verdict is 'pass' but "
                        "recommended_next_action is not "
                        "'accept'."
                    )
                )


        if (
            verdict
            ==
            "revise"
        ):

            if not issues:

                errors.append(
                    (
                        "Critic verdict is 'revise' "
                        "but no issues were reported."
                    )
                )


            if (
                recommended_action
                ==
                "accept"
            ):

                errors.append(
                    (
                        "Critic verdict is 'revise' "
                        "but action is 'accept'."
                    )
                )


        # ----------------------------------------------------
        # Critic cannot invent replacement columns
        # ----------------------------------------------------

        unknown_tokens = (
            _critic_unknown_column_tokens(
                data,
                state,
            )
        )


        if unknown_tokens:

            errors.append(
                (
                    "Critic introduces unknown variable "
                    "names inside recommended fixes: "
                    f"{unknown_tokens}."
                )
            )


        # ----------------------------------------------------
        # Independent upstream deterministic audit
        # ----------------------------------------------------

        deterministic_findings = (
            _deterministic_upstream_audit(
                state
            )
        )


        required_pairs = {
            (
                finding[
                    "source_agent"
                ],
                finding[
                    "category"
                ],
            )
            for finding
            in deterministic_findings
        }


        observed_pairs = {
            (
                issue[
                    "source_agent"
                ],
                issue[
                    "category"
                ],
            )
            for issue
            in issues
        }


        if (
            deterministic_findings
            and
            verdict
            !=
            "revise"
        ):

            errors.append(
                (
                    "Deterministic upstream audit found "
                    "material problems, therefore Critic "
                    "must return 'revise'. Findings: "
                    +
                    json.dumps(
                        deterministic_findings
                    )
                )
            )


        missing_pairs = (
            required_pairs
            -
            observed_pairs
        )


        if missing_pairs:

            errors.append(
                (
                    "Critic failed to report deterministic "
                    "upstream issue categories: "
                    f"{sorted(missing_pairs)}."
                )
            )


        affected_agents = {
            finding[
                "source_agent"
            ]
            for finding
            in deterministic_findings
        }


        if (
            affected_agents
            ==
            {"planner"}
        ):

            expected_action = (
                "revise_planner"
            )


        elif (
            affected_agents
            ==
            {"statistician"}
        ):

            expected_action = (
                "revise_statistician"
            )


        elif (
            affected_agents
            ==
            {
                "planner",
                "statistician",
            }
        ):

            expected_action = (
                "revise_both"
            )


        else:

            expected_action = (
                "accept"
            )


        if (
            recommended_action
            !=
            expected_action
        ):

            errors.append(
                (
                    "Critic routing recommendation does "
                    "not match deterministic audit. "
                    f"Expected '{expected_action}', "
                    f"got '{recommended_action}'."
                )
            )


    # ========================================================
    # Reject all remaining semantic violations
    # ========================================================

    if errors:

        raise ValueError(
            "\n".join(
                f"- {error}"
                for error
                in errors
            )
        )