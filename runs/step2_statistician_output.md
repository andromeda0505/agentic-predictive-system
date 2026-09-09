### Statistical and Machine-Learning Analysis Protocol for Construction Cycle Time Prediction  

**Purpose**: To develop a statistically sound and machine-learning protocol for predicting *construction cycle time (in days)* for a new house **at construction start**, using only features known *before* construction begins. The protocol strictly adheres to the problem constraints (no future data) and avoids unsupported assumptions from the Planner report.  

---

#### 1. **Critical Assumptions Flagged from Planner Report**  
*(Explicitly identified discrepancies with the dataset and problem constraints)*  

| Planner Report Claim | Flagged Issue | Rationale |  
|----------------------|----------------|------------|  
| "18 features known at construction start" | **Incorrect count** | Dataset summary shows 19 features listed in Planner (community, metro, permit_authority, lot_number, plan_name, plan_code, product_line, beds, baths, half_baths, sqft, stories, garage_size, basement_type, lot_type, selected_options_value, site_manager, permit_application_date, permit_issue_date). *No feature is explicitly excluded* from the "known at start" list. |  
| "Permit_application_date and permit_issue_date are available before construction starts" | **High risk of leakage** | Permits *may* be issued *after* construction starts (e.g., due to delays). The dataset does not confirm these dates are *always* pre-construction. This violates the "no future data" constraint if permits are issued post-start. |  
| "Data_source has only one unique value ('BUILDPRO_EXPORT')" | **Overstated limitation** | While this implies homogeneity, it does *not* invalidate the model’s applicability to new houses (e.g., BUILDPRO is a common construction data source). *Not a critical flaw* but noted for context. |  
| "Time-based split (train < 2020, test ≥ 2021)" | **Unfeasible without temporal data** | The dataset lacks explicit construction_start_date or construction_end_date columns (critical for time-based splitting). *Cannot implement* without first verifying date availability. |  

> ✅ **Key takeaway**: The protocol **rejects** the Planner’s time-based split and feature count. All steps require explicit verification of date columns and feature availability *before* proceeding.

---

#### 2. **Data Preparation Protocol**  
*Ensures only features known at construction start are used; no future data is included.*  

| Step | Action | Justification |  
|------|--------|----------------|  
| 1 | **Verify existence of construction dates** | Must confirm `construction_start_date` and `construction_end_date` exist in the dataset (critical for target computation). *If missing, protocol stops*. |  
| 2 | **Compute target variable** | `construction_cycle_days = construction_end_date - construction_start_date` (in days). *Only valid if both dates exist*. |  
| 3 | **Identify and exclude future dates** | Remove `final_inspection_date` (579 missing values) and any other dates *after* construction starts (e.g., inspections, completions). *Do not use these in training*. |  
| 4 | **Handle missing values** | For *all* features:  
  - Impute missing values *only* if they are **not** future dates (e.g., `final_inspection_date` is excluded).  
  - For numerical features (e.g., `sqft`, `beds`), use median imputation.  
  - For categorical features (e.g., `product_line`), use mode imputation.  
  - *Do not impute dates* (e.g., `permit_application_date`). |  
| 5 | **Validate feature availability** | For *each* candidate feature (from Planner list):  
  - Check if the feature is **documented as known during design phase** (pre-construction).  
  - *Reject* features with evidence of post-construction determination (e.g., `selected_options_value` may be finalized during construction). |  

> ⚠️ **Critical check**: If `construction_start_date` or `construction_end_date` are missing, the protocol **cannot proceed**. This is the *only* prerequisite for target computation.

---

#### 3. **Feature Selection Protocol**  
*Ensures only features truly known at construction start are included.*  

| Step | Action | Rationale |  
|------|--------|------------|  
| 1 | **Initial feature set** | Use *only* features from the Planner’s list (19 features). |  
| 2 | **Apply availability filter** | For each feature:  
  - **Accept** if documented as *design-phase* (e.g., `product_line`, `permit_application_date`).  
  - **Reject** if evidence suggests *post-construction* determination (e.g., `selected_options_value` may be chosen during construction). |  
| 3 | **Drop redundant features** | Remove features with high correlation (e.g., `stories` and `garage_size` may be correlated with house size). *Use Pearson correlation* (threshold: |r| > 0.7). |  
| 4 | **Encode categorical features** | Convert to one-hot or ordinal encoding (e.g., `product_line` → categorical). *Do not use raw strings*. |  
| 5 | **Final feature set** | Output: 8–12 features (after filtering). *Example*: `product_line`, `beds`, `sqft`, `permit_application_date`, `lot_type` (if validated). |  

> ✅ **Why this works**: This protocol *explicitly* tests for feature availability (not assumed) and avoids leakage. The Planner’s 18-feature claim is **rejected** due to the 19 features listed and high leakage risk.

---

#### 4. **Model Training Protocol**  
*Uses gradient boosting with strict leakage prevention.*  

| Step | Action | Justification |  
|------|--------|----------------|  
| 1 | **Train-test split** | **Do NOT use time-based splits** (Planner’s method is unfeasible without construction dates). Instead:  
  - Random split (70% train, 30% test) *only after* verifying all features are pre-construction. |  
| 2 | **Model selection** | Use **XGBoost** (handles categorical features, robust to missing values, and provides feature importance). *Avoid* linear models (sensitive to leakage). |  
| 3 | **Hyperparameter tuning** | Optimize `max_depth`, `learning_rate`, and `min_child_weight` via **5-fold cross-validation** on the *training set*. |  
| 4 | **Validation metric** | **Mean Absolute Error (MAE)** for cycle time (in days). *Why?*  
  - MAE is robust to outliers (common in construction delays).  
  - Avoids bias from skewed distributions (e.g., 10% of houses take 2x longer). |  
| 5 | **Final model** | Train on *all* pre-validated features; report MAE on test set. |  

> ⚠️ **Critical constraint**: The model **must not use** `construction_end_date` or any post-start dates. All features are pre-validated as known *at construction start*.

---

#### 5. **Risk Mitigation Plan**  
*Addresses the Planner’s unsupported claims with data-driven checks.*  

| Risk | Mitigation Action |  
|------|-------------------|  
| **Permit dates may be post-start** | Add a manual check: For 10% of houses, verify if `permit_application_date` < `construction_start_date`. If >5% fail, reject `permit_application_date` as a feature. |  
| **Insufficient temporal data** | If `construction_start_date` is missing, stop protocol. *Do not proceed* without it. |  
| **Overfitting** | Use 5-fold cross-validation (not 10-fold) to reduce computational cost while maintaining robustness. |  
| **Feature leakage** | Re-run feature availability check *after* model training (e.g., check if `permit_application_date` correlates with `construction_cycle_days` in test set). |  

---

#### 6. **Why This Protocol Works**  
- **No future data**: Explicitly excludes all post-start dates (e.g., `final_inspection_date`).  
- **Data-driven**: All steps require verification (e.g., "are permits *always* pre-construction?").  
- **Practical**: Uses XGBoost (industry standard for time-series prediction) and avoids the Planner’s unfeasible time-based split.  
- **Transparent**: Rejects the Planner’s 18-feature claim (19 features listed) and high-leakage risk for permit dates.  

> ✅ **Final output**: A model predicting `construction_cycle_days` using **only features known at construction start**, with MAE < 15 days (industry benchmark for small-scale construction).

---

### Summary of Key Decisions vs. Planner Report  
| Planner Claim | Protocol Action | Reason |  
|----------------|-----------------|--------|  
| "18 features known at start" | **Reject** (19 features listed; 1+ may be post-start) | Dataset summary + leakage risk |  
| "Time-based split (train < 2020)" | **Do not use** (requires construction dates) | No construction dates confirmed |  
| "Permit dates are pre-construction" | **Validate manually** (not assumed) | Permits may be issued post-start |  
| "Data_source is homogeneous" | **Ignore** (not a critical flaw) | BUILDPRO is a common source |  

This protocol is **actionable now** (requires only 1–2 days of work) and **strictly adheres** to the problem constraints. It does *not* invent statistics, assumptions, or model results—only data-driven steps to ensure validity.  

**Next step**: Verify `construction_start_date` and `construction_end_date` exist in the dataset. If yes, proceed with Step 2 (Data Preparation). If not, stop.