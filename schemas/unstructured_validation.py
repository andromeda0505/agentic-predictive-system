from __future__ import annotations

import re
from typing import Any


# ============================================================
# Generic helpers
# ============================================================

def _rule_as_dict(
    rule: Any,
) -> dict[str, Any]:

    if hasattr(
        rule,
        "model_dump",
    ):
        return rule.model_dump()

    return dict(
        rule
    )


# ============================================================
# Researcher semantic validation
# ============================================================

def _validate_researcher(
    content: str,
    state: dict[str, Any],
) -> list[str]:
    """
    Validate Researcher advisory output.

    The Researcher may raise hypotheses and verification
    questions.

    It may NOT manufacture external empirical evidence or
    contradict deterministic evidence.
    """

    errors: list[str] = []

    text = str(
        content
    )

    lower = (
        text
        .lower()
    )


    # ========================================================
    # 1. No percentages
    # ========================================================

    percentage_matches = re.findall(
        r"\b\d+(?:\.\d+)?\s*%",
        text,
    )


    if percentage_matches:

        errors.append(
            (
                "RESEARCHER_PERCENTAGE_CLAIM: "
                "The memo contains unsupported percentage "
                "claims. Researcher must use authoritative "
                "counts instead of percentage estimates."
            )
        )


    # ========================================================
    # 2. No unsupported external evidence
    # ========================================================

    forbidden_external_phrases = [
        "industry standard",
        "industry standards",
        "industry data",
        "industry benchmark",
        "industry benchmarks",
        "as industry data shows",
        "industry-wide",
        "industry wide",
        "research shows",
        "studies show",
        "according to industry",
    ]


    if any(
        phrase
        in lower

        for phrase
        in forbidden_external_phrases
    ):

        errors.append(
            (
                "RESEARCHER_EXTERNAL_EVIDENCE: "
                "The memo presents external evidence that "
                "was not supplied to this node. Convert "
                "such claims into NEEDS VERIFICATION."
            )
        )


    # ========================================================
    # 3. No unsupported directional consequences
    # ========================================================

    unsupported_certainty_phrases = [
        "will overestimate",
        "will underestimate",
        "will systematically overestimate",
        "will systematically underestimate",
        "deployment will fail",
        "will produce unreliable predictions",
        "will produce inaccurate predictions",
        "financial loss",
        "financial losses",
        "safety risk",
        "safety risks",
        "missed opportunities",
    ]


    if any(
        phrase
        in lower

        for phrase
        in unsupported_certainty_phrases
    ):

        errors.append(
            (
                "RESEARCHER_UNSUPPORTED_CONSEQUENCE: "
                "The memo asserts a directional or business "
                "consequence that deterministic evidence "
                "has not established. Express it as a "
                "possible risk requiring verification."
            )
        )


    # ========================================================
    # 4. No external permit-process generalization
    # ========================================================

    unsupported_permit_phrases = [
        "permits can be issued after construction starts",
        "permits can be issued after construction begins",
        "permits may be issued after construction starts in practice",
        "permits may be issued after construction begins in practice",
    ]


    if any(
        phrase
        in lower

        for phrase
        in unsupported_permit_phrases
    ):

        errors.append(
            (
                "RESEARCHER_EXTERNAL_PERMIT_CLAIM: "
                "The memo generalizes beyond supplied "
                "permit-timing evidence. Restrict the claim "
                "to this dataset and mark production-process "
                "behavior NEEDS VERIFICATION."
            )
        )


    # ========================================================
    # 5. Deterministic censoring facts
    # ========================================================

    dataset_facts = state.get(
        "dataset_facts",
        {},
    )


    in_progress_missing = int(
        dataset_facts
        .get(
            "end_date_missingness_by_status",
            {},
        )
        .get(
            "In Progress",
            {},
        )
        .get(
            "missing_end_dates",
            0,
        )
    )


    if (
        in_progress_missing
        >
        0
    ):

        contradiction_phrases = [
            "no quantification of right-censored",
            "right-censored cases are not quantified",
            "count of right-censored cases is unknown",
            "the count of right-censored cases is unknown",
            "count of homes with construction_end_date missing is unknown",
        ]


        if any(
            phrase
            in lower

            for phrase
            in contradiction_phrases
        ):

            errors.append(
                (
                    "RESEARCHER_CENSORING_FACT_CONTRADICTION: "
                    "The memo says censoring counts are "
                    "unknown even though authoritative state "
                    "already supplies the relevant counts."
                )
            )


    # ========================================================
    # 6. Do not ask for a count already available
    # ========================================================

    if (
        in_progress_missing
        >
        0
        and
        "count of homes with construction_end_date missing"
        in lower
        and
        (
            "evidence needed"
            in lower
            or
            "most urgent gap"
            in lower
        )
    ):

        errors.append(
            (
                "RESEARCHER_REDUNDANT_EVIDENCE_REQUEST: "
                "The memo requests a censoring count that "
                "is already present in authoritative state."
            )
        )


    # ========================================================
    # 7. No invented operational effect size
    # ========================================================

    invented_effect_patterns = [
        r"increase(?:s)?\s+(?:the\s+)?cycle\s+by",
        r"decrease(?:s)?\s+(?:the\s+)?cycle\s+by",
        r"reduce(?:s)?\s+(?:the\s+)?cycle\s+by",
        r"add(?:s)?\s+\d+\s+days",
        r"reduce(?:s)?\s+\d+\s+days",
    ]


    if any(
        re.search(
            pattern,
            lower,
        )

        for pattern
        in invented_effect_patterns
    ):

        errors.append(
            (
                "RESEARCHER_INVENTED_EFFECT_SIZE: "
                "The memo assigns an operational effect "
                "without deterministic evidence. Ask "
                "whether the effect exists instead."
            )
        )


    return errors


# ============================================================
# Sanitized retry feedback
# ============================================================

def researcher_retry_feedback(
    validation_error: str,
) -> list[str]:
    """
    Convert validator output into SAFE retry instructions.

    Critical design principle:

    Do NOT feed the rejected percentages or phrases back to
    the LLM verbatim.

    Otherwise a small deterministic model may simply echo the
    invalid text.
    """

    lower = (
        str(
            validation_error
        )
        .lower()
    )


    instructions: list[str] = []


    if (
        "researcher_percentage_claim"
        in lower
    ):

        instructions.append(
            (
                "Remove every percentage and every percent "
                "sign from the memo. Use only deterministic "
                "counts supplied in authoritative state."
            )
        )


    if (
        "researcher_external_evidence"
        in lower
    ):

        instructions.append(
            (
                "Remove all claims based on outside "
                "industry knowledge, benchmarks, studies, "
                "or external data. Convert unresolved ideas "
                "to NEEDS VERIFICATION questions."
            )
        )


    if (
        "researcher_unsupported_consequence"
        in lower
    ):

        instructions.append(
            (
                "Remove deterministic directional or "
                "business consequences. Use cautious "
                "phrasing such as MAY CREATE RISK or "
                "NEEDS VERIFICATION."
            )
        )


    if (
        "researcher_external_permit_claim"
        in lower
    ):

        instructions.append(
            (
                "Do not generalize permit behavior outside "
                "the supplied dataset. Describe future "
                "production behavior as NEEDS VERIFICATION."
            )
        )


    if (
        "researcher_censoring_fact_contradiction"
        in lower
    ):

        instructions.append(
            (
                "Use the censoring counts already supplied "
                "in authoritative dataset facts. Do not "
                "describe them as unknown."
            )
        )


    if (
        "researcher_redundant_evidence_request"
        in lower
    ):

        instructions.append(
            (
                "Do not request deterministic counts that "
                "are already supplied. Focus instead on "
                "how the censored observations should be "
                "handled."
            )
        )


    if (
        "researcher_invented_effect_size"
        in lower
    ):

        instructions.append(
            (
                "Remove all invented operational effect "
                "sizes. Ask whether such effects exist."
            )
        )


    if not instructions:

        instructions.append(
            (
                "The memo failed deterministic grounding "
                "validation. Re-read authoritative evidence "
                "and remove every unsupported factual claim."
            )
        )


    return instructions


# ============================================================
# Deterministic Researcher fallback
# ============================================================

def build_researcher_safe_fallback(
    state: dict[str, Any],
) -> str:
    """
    Construct a deterministic Researcher memo entirely from
    authoritative state.

    This is used only after the LLM repeatedly fails
    grounding validation.

    It guarantees that an advisory agent cannot crash the
    whole workflow merely because the local LLM cannot obey
    grounding constraints.
    """

    dataset_facts = state.get(
        "dataset_facts",
        {},
    )


    target_spec = dataset_facts.get(
        "target_specification",
        {},
    )


    status_counts = dataset_facts.get(
        "construction_status_counts",
        {},
    )


    missingness_by_status = dataset_facts.get(
        "end_date_missingness_by_status",
        {},
    )


    deterministic_checks = dataset_facts.get(
        "deterministic_checks",
        {},
    )


    target_name = target_spec.get(
        "name",
        "construction_cycle_days",
    )


    problem_type = target_spec.get(
        "problem_type",
        "regression",
    )


    prediction_time = target_spec.get(
        "prediction_time",
        "construction_start",
    )


    valid_target_count = target_spec.get(
        "valid_derived_target_count",
        "unknown",
    )


    invalid_duration_count = target_spec.get(
        "invalid_negative_duration_count",
        "unknown",
    )


    complete_count = status_counts.get(
        "Complete",
        "unknown",
    )


    in_progress_count = status_counts.get(
        "In Progress",
        "unknown",
    )


    in_progress_missing_end = (
        missingness_by_status
        .get(
            "In Progress",
            {},
        )
        .get(
            "missing_end_dates",
            "unknown",
        )
    )


    exact_duplicate_count = (
        deterministic_checks.get(
            "exact_duplicate_row_count",
            "unknown",
        )
    )


    permit_after_start_count = (
        deterministic_checks.get(
            "permit_issue_after_start_count",
            "unknown",
        )
    )


    # ========================================================
    # Verify-only fields
    # ========================================================

    verify_fields: list[str] = []


    for (
        feature_name,
        rule,
    ) in state.get(
        "feature_policy",
        {},
    ).items():

        rule_dict = (
            _rule_as_dict(
                rule
            )
        )


        if (
            rule_dict.get(
                "decision"
            )
            ==
            "verify"
        ):

            verify_fields.append(
                feature_name
            )


    verify_fields = sorted(
        verify_fields
    )


    verify_text = (
        ", ".join(
            verify_fields
        )
        if verify_fields
        else
        "none"
    )


    # ========================================================
    # Build memo
    # ========================================================

    return f"""EVIDENCE CHECK

ESTABLISHED: The authoritative target is {target_name}.
ESTABLISHED: The problem type is {problem_type}.
ESTABLISHED: Prediction occurs at {prediction_time}.
ESTABLISHED: There are {valid_target_count} valid derived target observations.
ESTABLISHED: There are {invalid_duration_count} invalid negative target durations.
ESTABLISHED: There are {complete_count} Complete observations and {in_progress_count} In Progress observations.
ESTABLISHED: {in_progress_missing_end} In Progress observations have missing construction_end_date and may represent right-censored outcomes.

PREDICTION-TIME RISKS

ESTABLISHED: construction_end_date and other blocked future fields must not be used as prediction-time predictors.
NEEDS VERIFICATION: Confirm production-time availability for fields currently marked verify: {verify_text}.
ESTABLISHED: Historical permit data contain {permit_after_start_count} observed permit_issue_date values after construction_start_date.
NEEDS VERIFICATION: Confirm that the historical permit-timing relationship remains valid in the future production process.

BUSINESS-PROCESS QUESTIONS

NEEDS VERIFICATION: Confirm when each verify-status field becomes final relative to construction_start.
NEEDS VERIFICATION: Determine whether relevant operational processes change over time in ways that could affect transportability.
NEEDS VERIFICATION: Determine whether additional operational information exists and is available at prediction time before proposing it as a feature.

DATA AND DEPLOYMENT RISKS

ESTABLISHED: There are {exact_duplicate_count} exact duplicate rows requiring deterministic handling.
ESTABLISHED: Missing construction_end_date among unfinished homes is an outcome-censoring issue rather than ordinary predictor missingness.
NEEDS VERIFICATION: Evaluate sensitivity of completed-case regression to exclusion of unfinished observations.
NEEDS VERIFICATION: Compare a censoring-aware survival-analysis candidate with completed-case regression.

PLANNER REVIEW

ESTABLISHED: The Planner should retain {target_name} as the regression target.
ESTABLISHED: Prediction-time feature decisions must follow the authoritative feature contract.
NEEDS VERIFICATION: The temporal holdout cutoff should be selected by a deterministic validation procedure rather than asserted without evidence.

RECOMMENDATIONS FOR THE CRITIC

ESTABLISHED: Audit target definition, leakage control, regression metrics, temporal validation, and censoring treatment against authoritative evidence.
NEEDS VERIFICATION: Require business-process confirmation before verify-status fields become production predictors.
NEEDS VERIFICATION: Require censoring-aware comparison before deciding how unfinished observations should contribute to model development.
"""


# ============================================================
# Public validator
# ============================================================

def validate_unstructured_output(
    node_name: str,
    content: str,
    state: dict[str, Any],
):
    """
    Validate an unstructured agent before publication.
    """

    errors: list[str] = []


    if (
        node_name
        ==
        "researcher"
    ):

        errors.extend(
            _validate_researcher(
                content,
                state,
            )
        )


    if errors:

        raise ValueError(
            "\n".join(
                f"- {error}"

                for error
                in errors
            )
        )