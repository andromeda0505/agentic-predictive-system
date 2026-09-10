# Agentic Predictive Modeling System

A local, file-driven multi-agent data-science workflow built with **LangGraph**, **Pydantic**, deterministic Python tools, YAML workflow definitions, and SQLite checkpointing.

The project started as a simple sequential LLM workflow and evolved into a more rigorous system with explicit state, prediction-time feature contracts, structured outputs, deterministic validation, critic-driven revision loops, parallel execution, and durable checkpoints.

## Why this project exists

A useful data-science agent should do more than generate plausible text. It should distinguish between:

- facts that can be computed deterministically,
- judgments that require statistical reasoning,
- claims that are not supported by available evidence,
- variables available at prediction time versus variables that leak future information,
- local output repair versus full workflow revision,
- temporary execution state versus durable workflow history.

This repository makes those distinctions explicit in code.

## Current architecture

```text
                         ┌── Researcher ─────┐
                         │                   │
Task + Data ─► Planner ──┤                   ├──► Critic ─► Accept / Revise
                         │                   │
                         └── Statistician ───┘
```

The workflow is compiled dynamically from YAML into LangGraph.

Implemented capabilities include:

- declarative workflow configuration,
- shared typed state,
- deterministic dataset profiling,
- prediction-time feature policy,
- structured Pydantic agent outputs,
- semantic output validation,
- deterministic repair of narrow hard-contract violations,
- bounded critic-driven revision loops,
- parallel fan-out / fan-in,
- tool execution logs,
- SQLite checkpoint persistence,
- persistent `thread_id` execution,
- pause / resume,
- checkpoint history inspection,
- historical replay,
- controlled counterfactual state branching.

## Core design principles

### 1. Separate facts from judgments

Deterministic facts should be computed with tools and stored in shared state. Examples include row counts, missingness, duplicates, date ordering, target availability, feature coverage, and censoring counts.

LLMs should reason **from** those facts rather than inventing them.

### 2. Define the reference time before choosing features

A variable is not safe merely because it exists in the dataset. For every project, define the prediction, decision, or intervention reference time and ask:

> Could this value have been known at that time?

This is the main defense against target leakage and post-treatment bias.

### 3. Use an explicit feature contract

The project uses decisions such as:

```text
allow
block
verify
identifier
reference_time
training_only
```

This prevents downstream agents from silently redefining feature eligibility.

### 4. Preserve authority ordering

A useful default hierarchy is:

```text
original task
    >
deterministic evidence
    >
feature / data contract
    >
validated upstream artifacts
    >
agent interpretation
```

### 5. Structured shape is not enough

Pydantic validates schema shape. It does not prove that an answer is statistically or semantically correct.

The workflow therefore adds deterministic semantic validation after structured parsing.

### 6. Use two levels of correction

```text
Local correction
    └── malformed / unsupported output from one node
        └── retry that node only

Graph-level revision
    └── substantive disagreement discovered by Critic
        └── rerun the required dependency path
```

Cheap problems should not trigger expensive workflow-wide reruns.

### 7. Parallelism requires a real join barrier

Independent agents can fan out in parallel, but downstream work should not run until all required branches are complete.

### 8. Persistence is different from exporting files

```text
state.json
    = human-readable export

SQLite checkpoints
    = resumable execution memory

thread_id
    = identity of one workflow execution lineage
```

## Repository layout

```text
agentic-predictive-system/
│
├── README.md
├── DATA_SCIENCE_PROJECT_CHECKLIST.md
│
├── inputs/
│   ├── task.md
│   ├── data.csv
│   └── feature_policy.yaml
│
├── workflows/
│   └── predictive_modeling.yaml
│
├── skills/
│   ├── planner.md
│   ├── researcher.md
│   ├── statistician.md
│   └── critic.md
│
├── schemas/
│   ├── state.py
│   ├── graph_state.py
│   ├── workflow.py
│   ├── output_validation.py
│   └── unstructured_validation.py
│
├── checkpointing.py
├── checkpoint_history.py
├── checkpoint_replay.py
├── checkpoint_counterfactual.py
├── langgraph_compiler.py
├── structured_runtime.py
├── tool_runtime.py
├── dataset_profile.py
├── run_step10.py
├── run_step11.py
│
└── runs/
    ├── checkpoints/
    └── step11/
```

`runs/` contains generated runtime output and normally should not be committed unless you intentionally want example artifacts.

## Shared state

The workflow state contains concepts such as:

```text
task
dataset_path
dataset_facts
feature_policy
artifacts
warnings
tool_log
revision_count
max_revisions
revision_feedback
revision_limit_reached
```

Parallel channels such as artifacts and logs use reducers so independent branches can safely contribute state.

## Running the persistent workflow

Start a new persistent thread:

```bash
python run_step11.py start construction-demo-001
```

Pause after Planner to test persistence:

```bash
python run_step11.py start construction-demo-001 --pause-after planner
```

Inspect status:

```bash
python run_step11.py status construction-demo-001
```

Resume:

```bash
python run_step11.py resume construction-demo-001
```

## Checkpoint history

```bash
python checkpoint_history.py history construction-demo-001
```

Inspect a real historical checkpoint:

```bash
python checkpoint_history.py inspect construction-demo-001 "CHECKPOINT_ID"
```

## Historical replay

```bash
python checkpoint_replay.py replay construction-demo-001 "CHECKPOINT_ID"
```

## Controlled counterfactual branch

Example: ask what the downstream workflow would do if zero critic revision rounds were permitted:

```bash
python checkpoint_counterfactual.py branch \
  construction-demo-001 \
  "PLANNER_CHECKPOINT_ID" \
  --max-revisions 0
```

The original historical checkpoints remain stored.

## Local checkpoint database

The default durable store is:

```text
runs/checkpoints/predictive_modeling.sqlite
```

Do not delete it while you still need resumable workflow history.

## Development philosophy

The system was intentionally built incrementally:

```text
prompt
  ↓
agent
  ↓
multiple agents
  ↓
shared state
  ↓
workflow DSL
  ↓
dynamic graph compiler
  ↓
deterministic tools
  ↓
structured outputs
  ↓
semantic validation
  ↓
critic
  ↓
bounded revisions
  ↓
parallel branches
  ↓
durable checkpointing
```

Each abstraction was introduced to solve an observed failure mode rather than added merely for complexity.

The companion **`DATA_SCIENCE_PROJECT_CHECKLIST.md`** captures the broader methodological lessons for predictive modeling, causal inference, experimentation, and agent-assisted analysis.

## What should be committed

Recommended:

```text
README.md
DATA_SCIENCE_PROJECT_CHECKLIST.md
*.py
skills/
schemas/
workflows/
inputs/task.md
inputs/feature_policy.yaml
```

Review carefully before committing:

```text
inputs/data.csv
runs/
*.sqlite
.env
credentials
API keys
local model caches
large generated outputs
```

Whether `inputs/data.csv` can be committed depends on ownership, privacy, licensing, and repository size.

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# Environment / secrets
.env
.env.*
*.key

# Generated workflow output
runs/

# SQLite runtime state
*.sqlite
*.sqlite3
*.db

# IDE / OS
.vscode/
.DS_Store
Thumbs.db
```

If `.vscode/` contains project settings you intentionally share, remove that line.

## Git workflow

Run these commands from the repository root:

```bash
git status
git add README.md DATA_SCIENCE_PROJECT_CHECKLIST.md
git commit -m "Add project README and data science checklist"
git push
```

## Central lesson

> **Do not ask a probabilistic model to enforce a deterministic contract when code can enforce it.**

Use agents for reasoning. Use tools for facts. Use schemas for structure. Use validators for contracts. Use graphs for orchestration. Use checkpoints for recoverability.

## License

Add the license appropriate for your intended use before making the repository broadly reusable.
