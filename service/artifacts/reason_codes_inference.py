
def get_reason_codes_single_input(single_input_dict,
                                  model, preprocessor, label_encoders,
                                  feature_names, shap_explainer):
    """
    Generate reason codes untuk single customer input (untuk portal web)

    Parameters:
    -----------
    single_input_dict : dict
        Dictionary berisi customer data (raw format, sebelum preprocessing)
    model : trained model
        Calibrated model
    preprocessor : fitted preprocessor
        StandardScaler yang sudah di-fit
    label_encoders : dict
        Dictionary label encoders untuk categorical features
    feature_names : list
        List nama fitur (setelah feature engineering)
    shap_explainer : SHAP explainer
        SHAP explainer object

    Returns:
    --------
    result : dict
        JSON-friendly dictionary berisi probability, prediction, dan reason codes
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    # STEP 1: Convert dict to DataFrame
    X_single = pd.DataFrame([single_input_dict])

    # STEP 2: Feature engineering (same as training)
    X_featured = X_single.copy()

    # HAPUS duration
    if 'duration' in X_featured.columns:
        X_featured = X_featured.drop('duration', axis=1)

    # Age groups
    X_featured['age_group'] = pd.cut(
        X_featured['age'],
        bins=[0, 25, 35, 50, 65, 100],
        labels=['young', 'adult', 'middle', 'senior', 'elderly']
    )

    # Previous contact success
    X_featured['prev_success'] = (X_featured['poutcome'] == 'success').astype(int)

    # Campaign intensity
    X_featured['campaign_intensity'] = pd.cut(
        X_featured['campaign'],
        bins=[0, 1, 3, 5, 100],
        labels=['low', 'medium', 'high', 'very_high']
    )

    # Economic indicators interaction
    if 'emp.var.rate' in X_featured.columns and 'cons.price.idx' in X_featured.columns:
        X_featured['economic_stability'] = (
            X_featured['emp.var.rate'] * X_featured['cons.price.idx']
        )

    # Month season
    month_to_season = {
        'jan': 'winter', 'feb': 'winter', 'mar': 'spring',
        'apr': 'spring', 'may': 'spring', 'jun': 'summer',
        'jul': 'summer', 'aug': 'summer', 'sep': 'fall',
        'oct': 'fall', 'nov': 'fall', 'dec': 'winter'
    }
    X_featured['season'] = X_featured['month'].map(month_to_season)

    # Label encode categorical columns
    categorical_cols = X_featured.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            X_featured[col] = X_featured[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # STEP 3: Ensure correct column order
    X_single_featured = X_featured[feature_names]

    # STEP 4: Preprocessing (scaling)
    X_single_scaled = preprocessor.transform(X_single_featured)
    X_single_scaled = pd.DataFrame(X_single_scaled, columns=feature_names)

    # STEP 5: Predict probability
    probability = model.predict_proba(X_single_scaled)[0, 1]
    prediction = int(model.predict(X_single_scaled)[0])

    # STEP 6: Risk level
    if probability < 0.3:
        risk_level = 'Low'
    elif probability < 0.6:
        risk_level = 'Medium'
    else:
        risk_level = 'High'

    # STEP 7: Calculate SHAP values
    # Get base estimator
    if hasattr(model, 'calibrated_classifiers_'):
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model

    shap_values_single = shap_explainer.shap_values(X_single_scaled)

    # Handle different SHAP formats
    if isinstance(shap_values_single, list):
        shap_values_single = shap_values_single[1][0]
    else:
        shap_values_single = shap_values_single[0]

    # STEP 8: Get top 5 features
    top_5_indices = np.argsort(np.abs(shap_values_single))[-5:][::-1]

    top_5_features = []
    reason_codes_list = []

    for feat_idx in top_5_indices:
        feature_name = feature_names[feat_idx]
        shap_value = shap_values_single[feat_idx]

        sign = '+' if shap_value > 0 else '-'
        reason_code = f"{sign} {feature_name}"

        top_5_features.append(feature_name)
        reason_codes_list.append(reason_code)

    # STEP 9: Build result dictionary
    result = {
        'customer_id': 'new_customer',
        'probability': float(probability),
        'prediction': int(prediction),
        'prediction_label': 'Yes' if prediction == 1 else 'No',
        'risk_level': risk_level,
        'top_5_features': top_5_features,
        'reason_codes': ', '.join(reason_codes_list)
    }

    return result
