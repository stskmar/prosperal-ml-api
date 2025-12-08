# Model Artifacts - Bank Marketing Lead Scoring v2.0

**Generated:** 2025-12-04 13:18:11
**Model:** CatBoost
**Version:** v2.0
**Test ROC-AUC:** 0.8046
**Test PR-AUC:** 0.4523

**New in v2.0:**
- Cost Matrix Simulation untuk business profit optimization
- Reason Codes (Top-5 SHAP) untuk explainability
- Single input inference function untuk portal web

---

## 📦 Files Included

1. **CatBoost_calibrated_model.pkl** - Trained & calibrated model
2. **preprocessor.pkl** - Preprocessing pipeline (StandardScaler)
3. **label_encoders.pkl** - Label encoders untuk categorical features
4. **feature_names.pkl** - List of feature names (maintain column order)
5. **feature_importance.pkl** - SHAP feature importance DataFrame
6. **feature_importance.csv** - SHAP feature importance (CSV)
7. **shap_explainer.pkl** - SHAP explainer untuk reason codes (NEW)
8. **cost_matrix_results.pkl** - Cost matrix simulation results (NEW)
9. **reason_codes_inference.py** - Python function untuk inference (NEW)
10. **metadata.pkl** - Model metadata (pickle)
11. **metadata.json** - Model metadata (JSON)
12. **README.md** - This file

---

## 🚀 How to Use - Single Input Inference (Portal Web)
```python
import pickle
import pandas as pd

# ============================================================================
# STEP 1: LOAD ALL ARTIFACTS
# ============================================================================

with open('outputs/model_artifacts/CatBoost_calibrated_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('outputs/model_artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('outputs/model_artifacts/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('outputs/model_artifacts/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('outputs/model_artifacts/shap_explainer.pkl', 'rb') as f:
    shap_explainer = pickle.load(f)

print("✓ All artifacts loaded")

# ============================================================================
# STEP 2: IMPORT INFERENCE FUNCTION
# ============================================================================

# Option 1: Import from reason_codes_inference.py
from outputs.model_artifacts.reason_codes_inference import get_reason_codes_single_input

# Option 2: Copy-paste function definition here (see file for full code)

# ============================================================================
# STEP 3: PREPARE SINGLE CUSTOMER INPUT
# ============================================================================

sample_customer = {
    'age': 35,
    'job': 'technician',
    'marital': 'married',
    'education': 'university.degree',
    'default': 'no',
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'month': 'may',
    'day_of_week': 'mon',
    'campaign': 2,
    'pdays': 999,
    'previous': 0,
    'poutcome': 'nonexistent',
    'emp.var.rate': 1.1,
    'cons.price.idx': 93.994,
    'cons.conf.idx': -36.4,
    'euribor3m': 4.857,
    'nr.employed': 5191.0
}

# ============================================================================
# STEP 4: GET PREDICTION + REASON CODES
# ============================================================================

result = get_reason_codes_single_input(
    single_input_dict=sample_customer,
    model=model,
    preprocessor=preprocessor,
    label_encoders=label_encoders,
    feature_names=feature_names,
    shap_explainer=shap_explainer
)

# ============================================================================
# STEP 5: USE RESULT (JSON-FRIENDLY)
# ============================================================================

import json
print(json.dumps(result, indent=2))

# Output example:
# {
#   "customer_id": "new_customer",
#   "probability": 0.4523,
#   "prediction": 0,
#   "prediction_label": "No",
#   "risk_level": "Medium",
#   "top_5_features": ["feature1", "feature2", ...],
#   "reason_codes": "+ feature1, - feature2, + feature3, ..."
# }
```

---

## 💰 Cost Matrix Optimal Threshold

**Business Optimization Results:**
- **Optimal Threshold:** 0.10
- **Optimal Top-X%:** 27.6%
- **Max Expected Profit:** IDR 69,930,000

**Confusion Matrix at Optimal Threshold:**
- TP: 738
- FP: 1754
- FN: 320
- TN: 6231

**Recommendation:**
Target top 27.6% of leads untuk maximize profit.

---

## ⚙️ Model Specifications

- **Algorithm:** CatBoost
- **Version:** v2.0
- **Test ROC-AUC (Calibrated):** 0.8046
- **Test PR-AUC:** 0.4523
- **Features:** 19
- **Training Samples:** 36,168

---

## 📞 Support

- **Full Model Card:** `outputs/model_card/MODEL_CARD_v2.md`
- **Cost Matrix Results:** `outputs/evaluation/cost_matrix_profit_curve.png`
- **Reason Codes CSV:** `outputs/reason_codes/reason_codes_test.csv`

---

**Status:** ✅ Ready for Production (v2.0)
