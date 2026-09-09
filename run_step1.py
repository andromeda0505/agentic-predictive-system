from pathlib import Path

import pandas as pd
import yaml
from ollama import chat


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).parent

SKILL_PATH = ROOT / "skills" / "planner.md"
TASK_PATH = ROOT / "inputs" / "task.md"
DATA_PATH = ROOT / "inputs" / "data.csv"

RUNS_DIR = ROOT / "runs"

OUTPUT_PATH = RUNS_DIR / "step1_planner_output.md"
PROMPT_SNAPSHOT_PATH = RUNS_DIR / "step1_prompt_snapshot.md"


# ============================================================
# Skill loader
# ============================================================

def load_skill(path: Path):
    """
    Load a Markdown skill containing YAML front matter.

    Expected structure:

    ---
    name: planner
    model: qwen3:4b
    ---

    Markdown instructions...
    """

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

    metadata_text = parts[1]
    instructions = parts[2].strip()

    metadata = yaml.safe_load(metadata_text)

    return metadata, instructions


# ============================================================
# Dataset summarizer
# ============================================================

def summarize_dataset(path: Path) -> str:
    """
    Create a compact machine-readable/human-readable summary
    of the dataset.

    We deliberately do NOT send the entire dataset to the LLM.
    """

    df = pd.read_csv(path)

    summary = []

    summary.append(f"Number of rows: {len(df):,}")
    summary.append(f"Number of columns: {len(df.columns):,}")
    summary.append("")

    summary.append("COLUMN SUMMARY")
    summary.append("=" * 80)

    for column in df.columns:

        series = df[column]

        dtype = str(series.dtype)

        missing_count = int(series.isna().sum())
        missing_pct = 100 * missing_count / len(df)

        unique_count = int(series.nunique(dropna=True))

        examples = (
            series
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

        summary.append(f"Column: {column}")
        summary.append(f"  dtype: {dtype}")
        summary.append(f"  missing: {missing_count:,} ({missing_pct:.2f}%)")
        summary.append(f"  unique values: {unique_count:,}")
        summary.append(f"  example values: {examples}")
        summary.append("")

    return "\n".join(summary)


# ============================================================
# Main execution
# ============================================================

def main():

    print("=" * 70)
    print("STEP 1 — SINGLE MARKDOWN SKILL → SINGLE LLM CALL")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    for required_file in [SKILL_PATH, TASK_PATH, DATA_PATH]:

        if not required_file.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_file}"
            )

    RUNS_DIR.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Load skill
    # --------------------------------------------------------

    metadata, skill_instructions = load_skill(SKILL_PATH)

    skill_name = metadata["name"]
    model = metadata["model"]

    print(f"\nLoaded skill: {skill_name}")
    print(f"Configured model: {model}")

    # --------------------------------------------------------
    # Load task
    # --------------------------------------------------------

    task = TASK_PATH.read_text(encoding="utf-8")

    print(f"Loaded task: {TASK_PATH.name}")

    # --------------------------------------------------------
    # Inspect dataset
    # --------------------------------------------------------

    print("Inspecting dataset...")

    dataset_summary = summarize_dataset(DATA_PATH)

    # --------------------------------------------------------
    # Construct user message
    # --------------------------------------------------------

    user_message = f"""
# Predictive Task

{task}

# Dataset Summary

{dataset_summary}

# Instruction

Using your skill definition and the information above,
create the predictive-modeling plan.

Do not fit a model.

Do not invent facts about columns that are not supported by
their names, types, or example values.

Explicitly distinguish observations from assumptions.
""".strip()

    # --------------------------------------------------------
    # Save exactly what the agent receives
    # --------------------------------------------------------

    prompt_snapshot = f"""
# SYSTEM / SKILL

{skill_instructions}

---

# USER INPUT

{user_message}
""".strip()

    PROMPT_SNAPSHOT_PATH.write_text(
        prompt_snapshot,
        encoding="utf-8"
    )

    print(
        f"Prompt snapshot saved to: "
        f"{PROMPT_SNAPSHOT_PATH.relative_to(ROOT)}"
    )

    # --------------------------------------------------------
    # Call local LLM
    # --------------------------------------------------------

    print(f"\nCalling local model: {model}")
    print("No external API key is being used.\n")

    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": skill_instructions,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        think=False,
        stream=False,
    )

    result = response.message.content

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    OUTPUT_PATH.write_text(
        result,
        encoding="utf-8"
    )

    print("=" * 70)
    print("PLANNER OUTPUT")
    print("=" * 70)
    print()
    print(result)
    print()
    print("=" * 70)

    print(
        f"\nResult saved to: "
        f"{OUTPUT_PATH.relative_to(ROOT)}"
    )

    print("\nSTEP 1 COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()