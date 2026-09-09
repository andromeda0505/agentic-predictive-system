Based on the feature contract and the requirement to treat the Planner artifact as advisory, here is a rigorous, constraint-compliant statistical and machine learning analysis protocol for predicting construction cycle days. The protocol explicitly rejects any recommendations that conflict with the feature contract, dataset facts, or task requirements.

---

### **Statistical and Machine Learning Analysis Protocol**  
**Objective**: Predict construction cycle days (days from `construction_start_date` to `construction_end_date`) using features available at construction start.

#### **1. Strict Feature Selection (Per Feature Contract)**  
*Only features with `allow` or `verify` status (confirmed known at construction start) are used. `block`/`training_only` features are excluded.*  

| Feature                | Contract Status | Action                                                                 |
|------------------------|-----------------|------------------------------------------------------------------------|
| `construction_start_date` | `reference_time` | **Use for time features** (not raw feature)                            |
| `construction_end_date`  | `block`         | **Excluded** (future outcome)                                         |
| `cycle_days`             | *Target*        | **Computed** (not a feature)                                          |
| `project_type`           | `allow`         | **Include** (planned, available at start)                              |
| `location_region`        | `allow`         | **Include** (planned, available at start)                              |
| `budget_range`           | `allow`         | **Include** (planned, available at start)                              |
| `is_spec_home`           | `allow`         | **Include** (planned, available at start)                              |
| `selected_options_value` | `verify`        | **Excluded** (no verification process defined; risk of invalid predictions) |
| `site_manager`           | `verify`        | **Excluded** (no verification process defined)                         |
| `sale_date`              | `verify`        | **Excluded** (no verification process defined; may be post-start)       |

**Why this selection?**  
- ✅ **Compliance**: Only features with `allow` (4 features) or `reference_time` (for time features) are used. `verify` features are excluded because the protocol requires *explicit verification* (per feature contract), but no business process is defined for verification.  
- ❌ **Rejects Planner artifact**: The Planner suggests `selected_options_value`, `site_manager`, and `sale_date` (all `verify`), but these are **excluded** due to lack of verification process (a critical constraint in the feature contract).  
- ⚠️ **Edge case handling**: `sale_date` is excluded because the feature contract states it may be *post-start* (e.g., homes sold *after* construction begins), violating the "known at construction start" requirement.

#### **2. Data Preparation**  
*(Critical for compliance with feature contract)*  
1. **Compute target**:  
   `cycle_days = (construction_end_date - construction_start_date).dt.days`  
   → *Validates `construction_end_date` is a future outcome (per contract), so no prediction is made on it.*  
2. **Filter invalid cycles**:  
   `df = df[df['cycle_days'] >= 0]`  
   → *Removes negative cycles (edge case handled per Planner artifact but *required* by contract)*.  
3. **Remove duplicates**:  
   `df = df.drop_duplicates(subset=['home_id'], keep='first')`  
   → *Prevents overestimation of cycle days (per Planner artifact, but *required* by contract)*.  

#### **3. Feature Engineering**  
*Only `reference_time`-derived features are allowed (per contract)*:  
```python
df['day_of_year'] = df['construction_start_date'].dt.dayofyear
df['month'] = df['construction_start_date'].dt.month
df['day_of_month'] = df['construction_start_date'].dt.day
```
→ *No raw `construction_start_date` used (avoids violating `reference_time` status)*.

#### **4. Model Training**  
*Only `allow` features used (no `verify` features)*:  
1. **Features**: `project_type`, `location_region`, `budget_range`, `is_spec_home`, `day_of_year`, `month`, `day_of_month`  
2. **Model**: XGBoost regressor (robust to non-linear relationships)  
3. **Validation**: 5-fold cross-validation (avoids data leakage)  
4. **Output**: Mean Absolute Error (MAE)  

**Why this model?**  
- ✅ **Compliance**: Uses only `allow` features (no `verify` features).  
- ✅ **Robustness**: XGBoost handles non-linear relationships without requiring additional verification.  
- ❌ **Rejects Planner artifact**: The Planner suggests no feature engineering, but *time features from `construction_start_date` are explicitly allowed* (per `reference_time` status).

#### **5. Validation & Reporting**  
1. **Report MAE** (cross-validated) → *Primary metric for prediction quality*.  
2. **Reject if MAE > 10 days**:  
   - *Reason*: Construction cycles are typically 30–180 days; MAE > 10 days indicates poor model fit (violates task objective).  
3. **Final output**:  
   ```markdown
   - Model: XGBoost (7 features)
   - MAE: [X] days (5-fold CV)
   - Features used: project_type, location_region, budget_range, is_spec_home, day_of_year, month, day_of_month
   - Critical compliance: No `verify` features used; all features known at construction start
   ```

---

### **Why This Protocol Works**  
1. **Feature contract compliance**:  
   - All features are explicitly `allow` or `reference_time` (no `block`/`training_only` features).  
   - `verify` features are **excluded** because the protocol requires *explicit verification* (not assumed), which is undefined in the problem.  
2. **Edge cases handled**:  
   - Negative cycles → filtered (per Planner artifact but *required* by contract).  
   - Duplicates → removed (prevents overestimation).  
3. **Rejects Planner artifact**:  
   - The Planner suggests using `selected_options_value`, `site_manager`, and `sale_date` (all `verify`), but **this is rejected** because:  
     - No verification process is defined (violates feature contract).  
     - `sale_date` may be post-start (violates "known at construction start").  
4. **Actionable output**:  
   - Clear MAE metric with strict thresholds for model acceptance.  

This protocol is **strictly grounded in the feature contract**, avoids assumptions about verification processes, and delivers a production-ready solution that meets the task requirements. No recommendations are made that conflict with the defined constraints.