import streamlit as st
import pandas as pd
import joblib
import json
import os

# Page configuration
st.set_page_config(
    page_title="E-commerce Price Prediction",
    page_icon="📦",
    layout="centered"
)

# Title and context
st.title("E-commerce Product Price Prediction System (ML Project)")
st.markdown(
    "Predicts product prices using a trained machine learning regression model "
    "based on historical e-commerce data."
)

# Load model and encoders
@st.cache_resource
def load_model_components():
    model_path = 'model/price_model.pkl'
    brand_path = 'model/brand_encoder.pkl'
    category_path = 'model/category_encoder.pkl'
    
    if not all(os.path.exists(p) for p in [model_path, brand_path, category_path]):
        st.error("❌ Model files missing. Run `python train_model.py` first.")
        st.stop()
    
    model = joblib.load(model_path)
    brand_encoder = joblib.load(brand_path)
    category_encoder = joblib.load(category_path)
    
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    
    return model, brand_encoder, category_encoder, metrics

model, brand_encoder, category_encoder, metrics = load_model_components()

# Display model performance metrics
st.subheader("Model Performance")
col1, col2 = st.columns(2)
xgboost_metrics = metrics.get('xgboost', {})
with col1:
    st.metric("RMSE (₹)", f"{xgboost_metrics.get('rmse', 0):.0f}")
with col2:
    st.metric("R² Score", f"{xgboost_metrics.get('r2', 0):.3f}")

# Fixed feature order
FEATURES = ["brand_enc", "category_enc", "discount"]

# User inputs
st.subheader("Product Details")
brand = st.text_input("Brand", value="Samsung")
category = st.text_input("Category", value="mobile")
discount = st.slider("Discount (%)", 0.0, 90.0, 15.0)

# Prediction
if st.button("Predict Price", type="primary"):
    try:
        # Safe encoding with fallback
        try:
            brand_enc = brand_encoder.transform([brand])[0]
        except:
            st.warning(f"⚠️ Brand '{brand}' not in training data. Using fallback.")
            brand_enc = 0
            
        try:
            category_enc = category_encoder.transform([category])[0]
        except:
            st.warning(f"⚠️ Category '{category}' not in training data. Using fallback.")
            category_enc = 0
        
        # Create feature vector
        input_data = pd.DataFrame({
            FEATURES[0]: [brand_enc],
            FEATURES[1]: [category_enc],
            FEATURES[2]: [discount]
        })
        
        # Predict
        predicted_price = model.predict(input_data)[0]
        
        # Display result
        st.success(f"**Predicted Price: ₹{predicted_price:,.0f}**")
        st.info(f"Brand: {brand} | Category: {category} | Discount: {discount}%")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Explainability
with st.expander("How Prediction Works"):
    st.markdown("""
    1. **Input Processing**: Brand/category encoded using pre-trained LabelEncoders
    2. **Feature Engineering**: [brand_enc, category_enc, discount]
    3. **Model Inference**: XGBoost regression predicts price instantly
    4. **Output**: Production-ready price prediction
    
    *Model trained offline via `train_model.py`*
    """)

st.markdown("---")
st.markdown("*Interview-ready ML deployment | No training in app.py*")

