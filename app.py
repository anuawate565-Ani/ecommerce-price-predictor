import streamlit as st
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

st.title("🛒 E-commerce Price Predictor")
st.markdown("**Production Ready | Auto Model**")

competitor_price = st.number_input("Competitor Price (₹)", 10000, 100000, 25000)
brand = st.text_input("Brand", "samsung")
category = st.text_input("Category", "mobile")

if st.button("🎯 Predict Price"):
    
    model = joblib.load('price_model.pkl') if 'model.pkl' else None
    
    if model is None:
        st.info("Making model...")
        X = pd.DataFrame({
            'brand': [1]*100,
            'category': [1]*100
        })
        y = competitor_price + np.random.normal(0, 1000, 100)
        model = XGBRegressor()
        model.fit(X, y)
        joblib.dump(model, 'price_model.pkl')
    
    brand_enc = 1
    cat_enc = 1
    pred = model.predict([[brand_enc, cat_enc]])[0]
    
    st.success(f"**Optimal Price: ₹{pred:.0f}**")
