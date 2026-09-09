### Predictive Modeling Plan for Construction Cycle Time

#### 1. **Problem Understanding**
- **Objective**: Predict the construction cycle time (in days) for a new house *at the moment construction begins*.
- **Key Constraint**: The prediction must be available *at construction start*, meaning we **cannot use future data** (e.g., `construction_end_date`). All features must be known *before* construction starts.
- **Data Source**: Historical dataset of completed houses (30 columns as described). The target `construction_cycle_days` is **not directly provided** but can be computed from `construction_start_date` and `construction_end_date`.

#### 2. **Target Variable Definition**
- **`construction_cycle_days`**: Number of days from `construction_start_date` to `construction_end_date`.
  - *Computation*:  
    `construction_cycle_days = (construction_end_date - construction_start_date).days`
  - *Why this works*: This is the only way to derive the target from the dataset (since the problem states the target is not precomputed).

#### 3. **Features Available at Construction Start**
The following features are **known and usable at construction start** (no future data required):
| Feature Name             | Type        | Why Available at Start?                                                                 |
|--------------------------|--------------|--------------------------------------------------------------------------------------|
| `community`              | Categorical  | Known during land acquisition (e.g., neighborhood).                                  |
| `metro`                  | Categorical  | Known during planning (e.g., city/region).                                           |
| `permit_authority`       | Categorical  | Known during permit application (e.g., city department).                            |
| `lot_number`             | Numerical    | Known during land survey.                                                           |
| `plan_name`              | Categorical  | Known during design phase.                                                          |
| `plan_code`              | Categorical  | Known during design phase.                                                          |
| `product_line`           | Categorical  | Known during design phase (e.g., "standard", "custom").                             |
| `beds`                   | Numerical    | Known during design phase.                                                          |
| `baths`                  | Numerical    | Known during design phase.                                                          |
| `half_baths`             | Numerical    | Known during design phase.                                                          |
| `sqft`                   | Numerical    | Known during design phase.                                                          |
| `stories`                | Numerical    | Known during design phase.                                                          |
| `garage_size`            | Numerical    | Known during design phase.                                                          |
| `basement_type`          | Categorical  | Known during design phase.                                                          |
| `lot_type`               | Categorical  | Known during land survey.                                                          |
| `selected_options_value` | Numerical    | Known during design phase (e.g., premium materials).                                |
| `site_manager`           | Categorical  | Known during construction planning.                                                 |
| `permit_application_date`| Date         | Known *before* construction starts (per permit process).                             |
| `permit_issue_date`     | Date         | Known *before* construction starts (per permit process).                             |

**Why these features work**:
- All features are **known at construction start** (no future data).
- `construction_start_date` is **not used as a feature** (it’s the reference point for cycle time; including it would be redundant since the target is *relative* to it).
- **Excluded features** (cannot be used at construction start):
  - `construction_end_date` (future data).
  - `construction_cycle_days` (target variable, not available until completion).

#### 4. **Model Training Process**
1. **Preprocess Data**:
   - Compute `construction_cycle_days` from `construction_start_date` and `construction_end_date`.
   - Convert categorical features (e.g., `community`, `metro`) to one-hot encodings or ordinal codes.
   - Handle missing values in numerical features (e.g., `sqft`) using median imputation.
   - Normalize numerical features (e.g., `beds`, `sqft`) to ensure model convergence.

2. **Feature Selection**:
   - Use **feature importance** (e.g., via random forest) to prioritize features that most strongly predict cycle time.
   - *Expected top features*: `permit_application_date` (time-related), `sqft` (size), `beds` (complexity), `product_line` (customization level).

3. **Model Selection**:
   - **Algorithm**: Gradient Boosting (e.g., XGBoost) or Random Forest.
   - *Why?* Handles mixed data types, captures non-linear relationships, and provides robust predictions for time-series-like construction cycles.
   - *Validation*: Split data into training (70%) and testing (30%) sets. Use **time-based splitting** (e.g., train on houses built before 2020, test on 2021+) to avoid leakage.

4. **Prediction Workflow**:
   ```mermaid
   graph LR
     A[New House] --> B{Features Available at Start}
     B -->|community| C
     B -->|metro| C
     B -->|permit_authority| C
     B -->|lot_number| C
     B -->|plan_name| C
     B -->|plan_code| C
     B -->|product_line| C
     B -->|beds| C
     B -->|baths| C
     B -->|half_baths| C
     B -->|sqft| C
     B -->|stories| C
     B -->|garage_size| C
     B -->|basement_type| C
     B -->|lot_type| C
     B -->|selected_options_value| C
     B -->|site_manager| C
     B -->|permit_application_date| C
     B -->|permit_issue_date| C
     C --> D[Model Prediction]
     D --> E[Construction Cycle Time in Days]
   ```

#### 5. **Why This Plan Works**
- **No Future Data Leakage**: All features are known *before* construction starts (critical for the problem constraint).
- **Realistic Target**: `construction_cycle_days` is derived from actual historical data (not assumed).
- **Practical for Construction**: Features align with real-world construction workflows (e.g., permits, design specs).
- **Scalable**: Works for new houses without requiring end-date data.

#### 6. **Example Prediction**
- **Input**: A new house with:
  - `community` = "Suburbia"
  - `sqft` = 2,500
  - `beds` = 3
  - `permit_application_date` = 2023-08-15
- **Output**: Model predicts `construction_cycle_days` = **120 days** (e.g., 4 months).

#### 7. **Key Risks & Mitigation**
| Risk                          | Mitigation Strategy                                  |
|-------------------------------|-----------------------------------------------------|
| Feature leakage (e.g., using `construction_end_date`) | Exclude all future dates; use time-based splits.     |
| Overfitting to historical data | Use cross-validation; add regularization (e.g., L2). |
| Unpredictable construction delays | Model only predicts *typical* cycles; add a confidence interval. |

This plan ensures the prediction is **actionable at construction start** while leveraging all available historical data without violating the problem constraints. 

**Final Answer**: The model predicts construction cycle time (in days) using **18 features known at construction start** (e.g., `community`, `sqft`, `permit_application_date`), computed from historical data where `construction_cycle_days` is derived from `construction_start_date` and `construction_end_date`.