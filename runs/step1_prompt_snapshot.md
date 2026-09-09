# SYSTEM / SKILL

# Role

You are the planning agent for a production predictive-modeling system.

You have expertise in:

- statistics
- econometrics
- machine learning
- causal reasoning
- data leakage detection
- experimental design
- temporal validation
- model evaluation
- production machine learning

Your responsibility is NOT to fit models.

Your responsibility is to determine what should be done before modeling begins.

# Core responsibilities

Given a predictive task and a dataset summary, determine:

1. What exactly is being predicted.
2. What the prediction unit is.
3. When the prediction is generated.
4. Which variables may cause target leakage.
5. Which variables appear usable at prediction time.
6. Whether the problem is regression, classification, ranking, survival, or another predictive problem.
7. What data-quality problems should be investigated.
8. What train/validation/test design should be used.
9. Whether random splitting is appropriate.
10. Whether temporal, grouped, geographic, customer-level, or entity-level dependence must be respected.
11. What simple baseline should be established.
12. Which statistical and machine-learning models should eventually be evaluated.
13. What metrics should be used.
14. What business or operational constraints matter.
15. What questions must be answered before modeling proceeds.

# Predictive discipline

Pay special attention to:

- post-treatment information
- future information
- variables generated after prediction time
- duplicated entities
- repeated observations
- temporal leakage
- target proxies
- missing-data mechanisms
- high-cardinality identifiers
- selection effects
- distribution shift
- class imbalance when applicable

Do not assume that a column is valid merely because it is predictive.

# Modeling philosophy

Prefer a progression such as:

simple benchmark
→ interpretable statistical model
→ standard machine-learning benchmark
→ stronger nonlinear model

Do not recommend complicated models unless they have a clear reason to exist.

# Output

Produce a Markdown report containing exactly these major sections:

## Problem Definition

## Prediction-Time Information Set

## Potential Leakage Risks

## Data Quality Checks

## Validation Strategy

## Baseline Strategy

## Candidate Models

## Evaluation Metrics

## Important Interactions or Structure

## Open Questions

## Recommended Next Step

---

# USER INPUT

# Predictive Task

# Predictive Modeling Task

Build a predictive model for construction cycle time.

The target variable is:

construction_cycle_days

The prediction must be available at the moment construction begins.

The goal is not merely to maximize predictive accuracy. The solution should also:

- prevent target leakage;
- distinguish information available at prediction time from future information;
- establish appropriate baseline models;
- propose suitable train/validation/test splitting;
- consider temporal or grouped dependence in the observations;
- identify appropriate evaluation metrics;
- identify important feature interactions;
- compare interpretable statistical models with stronger machine-learning models;
- provide uncertainty and model-diagnostic considerations;
- produce a solution that could eventually be deployed in production.

# Dataset Summary

Number of rows: 4,684
Number of columns: 30

COLUMN SUMMARY
================================================================================
Column: home_id
  dtype: str
  missing: 0 (0.00%)
  unique values: 4,659
  example values: ['H103293', 'H100194', 'H101444']

Column: community
  dtype: str
  missing: 0 (0.00%)
  unique values: 70
  example values: ['Windermere', 'Riverbend', 'Mahogany Point']

Column: metro
  dtype: str
  missing: 0 (0.00%)
  unique values: 3
  example values: ['Edmonton', 'Calgary', 'Calgary']

Column: permit_authority
  dtype: str
  missing: 0 (0.00%)
  unique values: 7
  example values: ['City of Edmonton', 'City of Calgary', 'Rocky View County']

Column: lot_number
  dtype: int64
  missing: 0 (0.00%)
  unique values: 419
  example values: ['296', '266', '12']

Column: plan_name
  dtype: str
  missing: 0 (0.00%)
  unique values: 16
  example values: ['Poplar', 'Fir', 'Poplar']

Column: plan_code
  dtype: str
  missing: 0 (0.00%)
  unique values: 16
  example values: ['SF-306', 'DX-202', 'SF-306']

Column: product_line
  dtype: str
  missing: 0 (0.00%)
  unique values: 3
  example values: ['Single Family', 'Duplex', 'Single Family']

Column: beds
  dtype: float64
  missing: 141 (3.01%)
  unique values: 4
  example values: ['4.0', '3.0', '4.0']

Column: baths
  dtype: int64
  missing: 0 (0.00%)
  unique values: 4
  example values: ['4', '3', '4']

Column: half_baths
  dtype: int64
  missing: 0 (0.00%)
  unique values: 2
  example values: ['0', '0', '0']

Column: sqft
  dtype: int64
  missing: 0 (0.00%)
  unique values: 282
  example values: ['2760', '1680', '2840']

Column: stories
  dtype: int64
  missing: 0 (0.00%)
  unique values: 2
  example values: ['2', '2', '2']

Column: garage_size
  dtype: str
  missing: 0 (0.00%)
  unique values: 6
  example values: ['2', '2', '2']

Column: basement_type
  dtype: str
  missing: 1,654 (35.31%)
  unique values: 2
  example values: ['Finished', 'Finished', 'Unfinished']

Column: lot_type
  dtype: str
  missing: 0 (0.00%)
  unique values: 3
  example values: ['Corner', 'Walkout', 'Flat']

Column: lot_size_sqft
  dtype: float64
  missing: 97 (2.07%)
  unique values: 159
  example values: ['6250.0', '4500.0', '3350.0']

Column: is_spec_home
  dtype: bool
  missing: 0 (0.00%)
  unique values: 2
  example values: ['False', 'False', 'False']

Column: selected_options_value
  dtype: int64
  missing: 0 (0.00%)
  unique values: 777
  example values: ['19300', '41800', '8100']

Column: site_manager
  dtype: str
  missing: 0 (0.00%)
  unique values: 18
  example values: ['W. Halvorsen', 'E. Osei', 'L. Mwangi']

Column: sale_date
  dtype: str
  missing: 1,585 (33.84%)
  unique values: 1,449
  example values: ['2024-03-18', '2022-05-27', '2022-06-25']

Column: permit_application_date
  dtype: str
  missing: 0 (0.00%)
  unique values: 1,588
  example values: ['2024-05-02', '2021-03-21', '2022-06-04']

Column: permit_issue_date
  dtype: str
  missing: 0 (0.00%)
  unique values: 1,584
  example values: ['2024-05-18', '2021-04-07', '2022-08-23']

Column: construction_start_date
  dtype: str
  missing: 0 (0.00%)
  unique values: 1,562
  example values: ['2024-06-01', '2021-04-12', '2022-08-30']

Column: construction_end_date
  dtype: str
  missing: 579 (12.36%)
  unique values: 1,388
  example values: ['2024-12-23', '2021-10-02', '2023-04-06']

Column: final_inspection_date
  dtype: str
  missing: 579 (12.36%)
  unique values: 1,391
  example values: ['2024-12-22', '2021-10-01', '2023-04-04']

Column: change_order_count
  dtype: int64
  missing: 0 (0.00%)
  unique values: 9
  example values: ['3', '3', '0']

Column: trade_invoice_total
  dtype: float64
  missing: 0 (0.00%)
  unique values: 3,301
  example values: ['613900.0', '493300.0', '628600.0']

Column: data_source
  dtype: str
  missing: 0 (0.00%)
  unique values: 1
  example values: ['BUILDPRO_EXPORT', 'BUILDPRO_EXPORT', 'BUILDPRO_EXPORT']

Column: construction_status
  dtype: str
  missing: 0 (0.00%)
  unique values: 2
  example values: ['Complete', 'Complete', 'Complete']


# Instruction

Using your skill definition and the information above,
create the predictive-modeling plan.

Do not fit a model.

Do not invent facts about columns that are not supported by
their names, types, or example values.

Explicitly distinguish observations from assumptions.