"""
Week 6 Capstone — House Price Prediction Engine
Applied ML Internship, Smart AI Lab (SEECS, NUST)

Loads the joblib pipeline saved by Week6_Model_Deployment.ipynb and serves
it through a small Streamlit UI. Run with:

    streamlit run app.py

Make sure house_price_pipeline.joblib sits in the same folder as this file.
"""

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered",
)

ARTIFACT_PATH = "house_price_pipeline.joblib"

# SAIL lab brand colors, used for the metric card accent below
NAVY = "#0B1F3A"
AMBER = "#F5A623"


# ------------------------------------------------------------------------
# Cached model loading — only runs once per session, not on every rerun
# ------------------------------------------------------------------------
@st.cache_resource
def load_pipeline(path: str):
    """Load the serialized scaler + model + feature order from disk."""
    return joblib.load(path)


try:
    pipeline = load_pipeline(ARTIFACT_PATH)
    scaler = pipeline["scaler"]
    model = pipeline["model"]
    feature_names = pipeline["feature_names"]
except FileNotFoundError:
    st.error(
        f"Couldn't find '{ARTIFACT_PATH}'. Run the Week 6 notebook first — "
        "it generates this file when it serializes the trained pipeline."
    )
    st.stop()
except Exception as e:
    st.error(f"Failed to load the model pipeline: {e}")
    st.stop()


# ------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------
st.title("🏠 House Price Prediction Engine")
st.write("Adjust the property details below to get an estimated market value.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input(
        "Total Area (Sq Ft)", min_value=300, max_value=10000, value=1500, step=50
    )
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)

with col2:
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=8, value=2)
    age = st.number_input(
        "Property Age (Years)", min_value=0, max_value=100, value=10, step=1
    )

st.markdown("---")

# ------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------
if st.button("Calculate Estimated Price", use_container_width=True):
    # Build the input frame with the exact column order the scaler expects —
    # this is exactly why we saved feature_names alongside the model.
    input_data = pd.DataFrame([[sqft, bedrooms, bathrooms, age]], columns=feature_names)

    scaled_input = scaler.transform(input_data)
    predicted_price = model.predict(scaled_input)[0]

    st.subheader("Valuation Summary")
    st.metric(label="Predicted Market Value", value=f"${predicted_price:,.2f}")

    with st.expander("View input details"):
        st.json(
            {
                "Square Feet": int(sqft),
                "Bedrooms": int(bedrooms),
                "Bathrooms": int(bathrooms),
                "Property Age": int(age),
            }
        )

    st.caption(
        "Estimate from a Random Forest model trained on historical sales data. "
        "Treat this as a ballpark figure, not an appraisal."
    )
