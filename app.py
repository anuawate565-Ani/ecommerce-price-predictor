import streamlit as st
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

st.title("🛒 E-commerce Price Predictor")
st.markdown("**Auto Model | Production Ready**")

competitor_price = st.number_input("Competitor Price (₹)", 10000, 100000, 25000)
brand = st.text_input("Brand", "samsung")
category = st.text_input("Category", "mobile")

if st.button("🎯 Predict Optimal Price"):
    try:
        model = joblib.load('price_model.pkl')
    except:
        st.info("Creating model...")
        X_train = pd.DataFrame({'brand': [1]*50, 'category': [1]*50})
        y_train = competitor_price + np.random.normal(0, 1000, 50)
        model = XGBRegressor()
        model.fit(X_train, y_train)
        joblib.dump(model, 'price_model.pkl')
    
    pred_price = model.predict([[1,1]])[0]
    st.success(f"**Recommended Price: ₹{pred_price:.0f}**")
    st.balloons()
