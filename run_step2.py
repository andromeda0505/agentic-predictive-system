from pathlib import Path

import pandas as pd
import yaml
from ollama import chat


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).parent

INPUTS_DIR = ROOT / "inputs"
SKILLS_DIR = ROOT / "skills"
RUNS_DIR = ROOT / "runs"

DATA_PATH = INPUTS_DIR / "data.csv"
TASK_PATH = INPUTS_DIR / "task.md"

PLANNER_SKILL_PATH = SKILLS_DIR / "planner.md"
STATISTICIAN_SKILL_PATH = SKILLS_DIR / "statistician.md"

PLANNER_OUTPUT_PATH = RUNS_DIR / "step2_planner_output.md"

STATISTICIAN_OUTPUT_PATH = (
    RUNS_DIR / "step2_statistician_output.md"
)


# ============================================================
# Skill Loader
# ============================================================

def load_skill(path: Path):

    text = path.read_text(encoding="utf-8")

    if not text.startswith("---"):
        raise ValueError(
            f"{path} does not contain YAML front matter."
        )

    parts = text.split("---", 2)

    if len(parts) != 3:
        raise ValueError(
            f"Could not parse YAML front matter in {path}."
        )

    metadata = yaml.safe_load(parts[1])
    instructions = parts[2].strip()

    return metadata, instructions


# ============================================================
# Dataset Summary
# ============================================================

def summarize_dataset(path: Path) -> str:

    df = pd.read_csv(path)

    lines = []

    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append("")

    for column in df.columns:

        s = df[column]

        missing_count = int(s.isna().sum())
        missing_pct = 100 * missing_count / len(df)

        unique_count = int(s.nunique(dropna=True))

        examples = (
            s
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

        lines.append(f"Column: {column}")
        lines.append(f"  dtype: {s.dtype}")
        lines.append(
            f"  missing: {missing_count:,} "
            f"({missing_pct:.2f}%)"
        )
        lines.append(
            f"  unique values: {unique_count:,}"
        )
        lines.append(
            f"  examples: {examples}"
        )
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Clean Model Output
# ============================================================

def clean_output(text: str) -> str:
    """
    Some local reasoning models may occasionally expose a
    thinking block even when thinking is disabled.

    We only retain the final answer.
    """

    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    return text.strip()


# ============================================================
# Generic Agent Runner
# ============================================================

def run_agent(
    skill_path: Path,
    user_message: str,
) -> str:

    metadata, instructions = load_skill(skill_path)

    name = metadata["name"]
    model = metadata["model"]

    print()
    print("=" * 70)
    print(f"RUNNING AGENT: {name.upper()}")
    print(f"MODEL: {model}")
    print("=" * 70)

    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": instructions,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        think=False,
        stream=False,
        options={
            "temperature": 0.2,
        },
    )

    result = clean_output(
        response.message.content
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 2")
    print("PLANNER → STATISTICIAN")
    print("=" * 70)

    RUNS_DIR.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    task = TASK_PATH.read_text(
        encoding="utf-8"
    )

    print("\nReading dataset...")

    dataset_summary = summarize_dataset(
        DATA_PATH
    )

    # ========================================================
    # AGENT 1 — PLANNER
    # ========================================================

    planner_message = f"""
# Predictive Task

{task}

# Dataset Summary

{dataset_summary}

# Assignment

Create the predictive-modeling plan.

Do not fit models.

Clearly distinguish established information from
assumptions and questions requiring verification.
""".strip()

    planner_output = run_agent(
        PLANNER_SKILL_PATH,
        planner_message,
    )

    PLANNER_OUTPUT_PATH.write_text(
        planner_output,
        encoding="utf-8",
    )

    print(
        "\nPlanner artifact saved:"
    )

    print(
        PLANNER_OUTPUT_PATH.relative_to(ROOT)
    )

    # ========================================================
    # AGENT 2 — STATISTICIAN
    # ========================================================

    statistician_message = f"""
# Original Predictive Task

{task}


# Dataset Summary

{dataset_summary}


# Planner Report

{planner_output}


# Assignment

Using the original task, dataset summary, and Planner
report, create the statistical and machine-learning
analysis protocol.

The Planner report is advisory, not authoritative.

If you identify a questionable assumption, contradiction,
or unsupported recommendation in the Planner report,
explicitly flag it rather than repeating it.

No models have been fitted yet.

Do not invent statistical findings.
""".strip()

    statistician_output = run_agent(
        STATISTICIAN_SKILL_PATH,
        statistician_message,
    )

    STATISTICIAN_OUTPUT_PATH.write_text(
        statistician_output,
        encoding="utf-8",
    )

    print(
        "\nStatistician artifact saved:"
    )

    print(
        STATISTICIAN_OUTPUT_PATH.relative_to(ROOT)
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 2 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Execution chain:")
    print()
    print("task + dataset")
    print("      ↓")
    print("   planner")
    print("      ↓")
    print("planner artifact")
    print("      ↓")
    print(" statistician")
    print("      ↓")
    print("statistical protocol")


if __name__ == "__main__":
    main()