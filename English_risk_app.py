# ==============================================================================
# File Name: app.py
# Description: YLS Recidivism Prediction Tool (English / ID Only Version)
# ==============================================================================

import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. App Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="YLS Prediction Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🛡️ YLS Prediction Table")

# ------------------------------------------------------------------------------
# 2. YLS Data Definitions (English Domains, ID-only Labels)
# ------------------------------------------------------------------------------
YLS_DOMAINS = {
    "1. Prior and Current Offenses/Dispositions": [
        {"id": "YLS_1a", "label": "YLS 1a"},
        {"id": "YLS_1b", "label": "YLS 1b"},
        {"id": "YLS_1c", "label": "YLS 1c"},
        {"id": "YLS_1d", "label": "YLS 1d"},
        {"id": "YLS_1e", "label": "YLS 1e"},
    ],
    "2. Family Circumstances/Parenting": [
        {"id": "YLS_2a", "label": "YLS 2a"},
        {"id": "YLS_2b", "label": "YLS 2b"},
        {"id": "YLS_2c", "label": "YLS 2c"},
        {"id": "YLS_2d", "label": "YLS 2d"},
        {"id": "YLS_2e", "label": "YLS 2e"},
        {"id": "YLS_2f", "label": "YLS 2f"},
    ],
    "3. Education/Employment": [
        {"id": "YLS_3a", "label": "YLS 3a"},
        {"id": "YLS_3b", "label": "YLS 3b"},
        {"id": "YLS_3c", "label": "YLS 3c"},
        {"id": "YLS_3d", "label": "YLS 3d"},
        {"id": "YLS_3e", "label": "YLS 3e"},
        {"id": "YLS_3f", "label": "YLS 3f"},
        {"id": "YLS_3g", "label": "YLS 3g"},
    ],
    "4. Peer Relations": [
        {"id": "YLS_4a", "label": "YLS 4a"},
        {"id": "YLS_4b", "label": "YLS 4b"},
        {"id": "YLS_4c", "label": "YLS 4c"},
        {"id": "YLS_4d", "label": "YLS 4d"},
    ],
    "5. Substance Abuse": [
        {"id": "YLS_5a", "label": "YLS 5a"},
        {"id": "YLS_5b", "label": "YLS 5b"},
        {"id": "YLS_5c", "label": "YLS 5c"},
        {"id": "YLS_5d", "label": "YLS 5d"},
        {"id": "YLS_5e", "label": "YLS 5e"},
    ],
    "6. Leisure/Recreation": [
        {"id": "YLS_6a", "label": "YLS 6a"},
        {"id": "YLS_6b", "label": "YLS 6b"},
        {"id": "YLS_6c", "label": "YLS 6c"},
    ],
    "7. Personality/Behavior": [
        {"id": "YLS_7a", "label": "YLS 7a"},
        {"id": "YLS_7b", "label": "YLS 7b"},
        {"id": "YLS_7c", "label": "YLS 7c"},
        {"id": "YLS_7d", "label": "YLS 7d"},
        {"id": "YLS_7e", "label": "YLS 7e"},
        {"id": "YLS_7f", "label": "YLS 7f"},
        {"id": "YLS_7g", "label": "YLS 7g"},
    ],
    "8. Attitudes/Orientation": [
        {"id": "YLS_8a", "label": "YLS 8a"},
        {"id": "YLS_8b", "label": "YLS 8b"},
        {"id": "YLS_8c", "label": "YLS 8c"},
        {"id": "YLS_8d", "label": "YLS 8d"},
        {"id": "YLS_8e", "label": "YLS 8e"},
    ]
}

ALL_FEATURES = []
# Create a dictionary for labels (ID -> "YLS 1a")
LABELS_DICT = {}
for items in YLS_DOMAINS.values():
    for item in items:
        ALL_FEATURES.append(item["id"])
        LABELS_DICT[item["id"]] = item["label"]

# ------------------------------------------------------------------------------
# 3. Load Model (.ubj support)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_ai_model():
    model = xgb.XGBClassifier()
    try:
        # Load the lightweight .ubj model
        model.load_model("yls_model.ubj")
    except Exception as e:
        return None, f"Model loading error: {e}"
    
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_ai_model()

if model is None:
    st.error("⚠️ Error: 'yls_model.ubj' not found. Please ensure the model file is in the directory.")
    st.stop()

# ------------------------------------------------------------------------------
# 4. Layout (Left 35% : Right 65%)
# ------------------------------------------------------------------------------
col_input, col_result = st.columns([0.35, 0.65])

# --- Left Panel: Input Checkboxes ---
user_inputs = {}

with col_input:
    st.markdown("### 📋 Input Items")
    
    # Display by Domain
    for domain_name, items in YLS_DOMAINS.items():
        # Display domain header with a blue left border
        st.markdown(
            f"<div style='border-left: 5px solid #007bff; padding-left: 8px; margin-top: 15px; font-weight: bold;'>{domain_name}</div>",
            unsafe_allow_html=True
        )
        
        for item in items:
            # Checkbox
            is_checked = st.checkbox(item["label"], key=item["id"])
            user_inputs[item["id"]] = 1 if is_checked else 0

# --- Right Panel: Results ---
with col_result:
    st.markdown("### 📊 Prediction Result")

    # Create Input DataFrame
    input_df = pd.DataFrame([user_inputs])
    valid_input_df = input_df[ALL_FEATURES]

    # Prediction
    prob = model.predict_proba(valid_input_df)[0][1]
    total_score = valid_input_df.sum(axis=1).values[0]

    # Risk Level Logic (Based on Excel sheet thresholds)
    if prob >= 0.71:
        color = "#dc3545" # Red
        text = "High Risk"
        bg = "#ffe6e6"
    elif prob >= 0.33:
        color = "#fd7e14" # Orange
        text = "Medium Risk"
        bg = "#fff3cd"
    elif prob >= 0.18:
        color = "#0d6efd" # Blue
        text = "Low Risk"
        bg = "#e7f1ff"
    else:
        color = "#198754" # Green
        text = "Lowest Risk"
        bg = "#e8f5e9"

    # Display Result with HTML (font size 24px)
    st.markdown(f"""
    <div style="
        border: 2px solid {color};
        border-radius: 8px;
        background-color: {bg};
        padding: 15px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <div style="color:{color}; font-weight:bold; font-size:18px; margin-bottom:5px;">{text}</div>
        <hr style="border-top: 1px solid {color}; margin: 5px 0;">
        <div style="font-size: 12px; color: #555;">Predicted Recidivism Probability</div>
        <div style="font-size: 24px; font-weight: bold; color: {color}; line-height: 1.2;">
            {prob * 100:.1f}%
        </div>
        <p style="font-size: 14px; color: #333; margin-top: 5px;">
            Total Score: <b>{int(total_score)}</b> / 42
        </p>
    </div>
    """, unsafe_allow_html=True)

    # SHAP Graph Display
    st.markdown("**【Factor Analysis (SHAP Waterfall Plot)】**")
    
    shap_values = explainer(valid_input_df)
    # Use ID labels (e.g., "YLS 1a") for the graph
    shap_values.feature_names = [LABELS_DICT.get(f, f) for f in ALL_FEATURES]

    # Graph Adjustment (Vertical layout)
    fig, ax = plt.subplots(figsize=(8, 12))
    
    try:
        # Display all 42 items
        shap.plots.waterfall(shap_values[0, :, 1], max_display=42, show=False)
    except:
        shap.plots.waterfall(shap_values[0], max_display=42, show=False)
    
    st.pyplot(fig)
