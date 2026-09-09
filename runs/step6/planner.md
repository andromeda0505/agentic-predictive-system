Based on the provided JSON data and the prediction-time feature contract, here is a rigorous, step-by-step predictive modeling plan for estimating **construction cycle days** (from `construction_start_date` to `construction_end_date`) at the time of `construction_start_date`. This plan strictly adheres to the constraints and insights from your dataset.

---

### **1. Critical Constraints & Insights from Your Data**
Before modeling, we must address these key observations from your JSON:
- ✅ **Only 4105 rows have valid `construction_end_date`** (out of 4684 total rows with `construction_start_date`).  
  → *This is the only dataset we can use for training/validating predictions*.
- ✅ **7 rows have negative cycle days** (invalid due to time calculation).  
  → *These are explicitly removed from the training set*.
- ✅ **579 rows are "in-progress"** (no `construction_end_date`).  
  → *These will be the target for future predictions*.
- ✅ **All features must be available at `construction_start_date`** (no future data).  
  → *No `construction_end_date` or derived metrics can be used as predictors*.
- ✅ **Home IDs are duplicated** (25 home IDs appear twice).  
  → *This is a known data pattern but does not violate the feature contract*.

**Key Takeaway**: We **cannot** use `construction_end_date` as a predictor (it’s the target). All modeling must rely **only** on features available *at `construction_start_date`*.

---

### **2. Step-by-Step Modeling Plan**
#### **Step 1: Data Preparation & Cleaning**
| Action | Why | Implementation |
|--------|-----|-----------------|
| Filter rows with `construction_end_date` | Only 4105 rows are valid for training | `df_train = df[df['construction_end_date'].notnull()]` |
| Remove 7 negative cycle days | Invalid time calculations | `df_train = df_train[df_train['cycle_days'] > 0]` |
| Verify all features are available at `construction_start_date` | No future data allowed | Cross-check features against `construction_start_date` timestamps |

**Output**: Clean training dataset with **4098 rows** (4105 - 7).

#### **Step 2: Feature Engineering (Only at `construction_start_date`)**
We **must not** use `construction_end_date` or derived metrics. All features must be computable *before* construction starts:
| Feature | Type | Why It’s Valid |
|---------|------|----------------|
| `construction_start_date` | Timestamp | Core input (no future data) |
| `day_of_week` | Categorical | Derived from `construction_start_date` |
| `month` | Categorical | Derived from `construction_start_date` |
| `quarter` | Categorical | Derived from `construction_start_date` |
| `year` | Categorical | Derived from `construction_start_date` |
| `community` | Categorical | Pre-existing location data |
| `metro` | Categorical | Pre-existing location data |
| `permit_authority` | Categorical | Pre-existing location data |
| `selected_options` | Categorical | Must be frozen at `construction_start_date` (business verification required) |
| `site_manager` | Categorical | Must be frozen at `construction_start_date` (business verification required) |

**Critical Business Check**:  
→ **Do `selected_options` and `site_manager` values change *after* `construction_start_date`?**  
  - If **yes**: Drop these features (they violate the "no future data" rule).  
  - If **no**: Keep them (they are frozen at `construction_start_date`).  
  *(This is the #1 unresolved question requiring business input)*

#### **Step 3: Model Selection & Training**
| Model | Why It’s Best | Why Not Others |
|-------|----------------|----------------|
| **Random Forest Regressor** | Handles non-linear relationships, robust to outliers, no feature scaling needed | Linear models (e.g., LR) underfit; Gradient Boosting overfits with small datasets |
| **Why?** | 4098 rows is small but sufficient for RF. RF works well with categorical features (via one-hot encoding) and time features. |

**Training Process**:
1. Split data into **80% training** (3278 rows) and **20% validation** (820 rows).
2. Encode categorical features (e.g., `community`, `month`, `quarter`).
3. Train RF regressor on `construction_start_date`-derived features → predict `cycle_days`.

#### **Step 4: Validation & Error Analysis**
| Metric | Target Value | Why It Matters |
|--------|---------------|----------------|
| **MAE** | < 15 days | Ensures predictions are practical (construction cycles rarely deviate by >15 days) |
| **RMSE** | < 20 days | Checks for large outliers (e.g., extreme weather delays) |
| **R²** | > 0.75 | Confirms model explains >75% of cycle variance |

**Critical Checks**:
- **Do duplicates (25 home IDs) cause bias?**  
  → If yes: Stratify training by home ID (25 groups) to prevent overfitting.
- **Are predictions for in-progress rows (579 rows) realistic?**  
  → Validate against historical data (e.g., if 80% of past cycles were 60–90 days, predictions should cluster there).

#### **Step 5: Deploying Predictions for In-Progress Rows**
For the **579 in-progress rows** (no `construction_end_date`):
1. Use the trained model to predict `cycle_days` at `construction_start_date`.
2. **Do NOT use `construction_end_date`** (it’s unavailable).
3. **Output**: Predicted cycle days (e.g., "Estimated completion: 72 days").

**Example Output**:
| Home ID | Predicted Cycle Days | Confidence Interval |
|---------|----------------------|---------------------|
| 12345   | 68                   | [62, 74]            |
| 67890   | 85                   | [79, 91]            |

---

### **3. Unresolved Questions Requiring Business Input**
These **must be answered before deployment**:
1. **Are `selected_options` and `site_manager` values frozen at `construction_start_date`?**  
   → *If not, drop these features to avoid violating the "no future data" rule*.
2. **What is the business meaning of duplicated home IDs?**  
   → *Do they represent the same construction project twice? If yes, how should we handle them?* (e.g., average predictions, drop duplicates).

> 💡 **Why these questions matter**:  
> - If `selected_options`/`site_manager` change *after* `construction_start_date`, using them as predictors would be **causally invalid** (they’re not available at the time of prediction).  
> - Duplicated home IDs could indicate data errors or multiple projects. If they’re the same project, averaging predictions might improve accuracy.

---

### **4. Why This Plan Works for Your Constraints**
| Your Constraint | How This Plan Addresses It |
|-----------------|----------------------------|
| **No future data** | All features derived *only* from `construction_start_date` |
| **4105 valid rows** | Uses 4098 rows (after cleaning) for training |
| **579 in-progress rows** | Predicts cycle days *before* `construction_end_date` is known |
| **Home ID duplicates** | Handles via stratification or business validation |
| **Feature contract** | Strictly uses features available at `construction_start_date` |

---

### **5. Expected Outcomes & Timeline**
| Phase | Timeline | Deliverable |
|-------|----------|--------------|
| Data cleaning | 1 day | 4098-row training dataset |
| Feature engineering | 2 days | Categorical features ready |
| Model training | 3 days | RF regressor with 80% training accuracy |
| In-progress predictions | Real-time | 579 predictions (with confidence intervals) |
| Monthly monitoring | Ongoing | Error rate, model drift checks |

**Final Output**: A production-ready model that predicts **construction cycle days** for *new projects* (579 rows) **at the time of `construction_start_date`**, with no use of future data.

---

### **Key Takeaway**
**Deploy this plan immediately** – it’s the *only* way to estimate construction cycles without violating your data constraints. The **critical business check** (whether `selected_options`/`site_manager` are frozen) is the only step that requires your team’s input. Once resolved, the model will be ready in **< 10 days**.

> 🔑 **Remember**: *The model predicts the future (cycle days) using only past data (at `construction_start_date`), not the target itself (which is `construction_end_date`)*. This is the core of your prediction-time contract.

Let me know if you need help with the business check for `selected_options`/`site_manager`! I can draft a simple questionnaire for your team.