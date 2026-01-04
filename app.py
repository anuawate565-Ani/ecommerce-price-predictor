import streamlit as st
import numpy as np

st.title("🛒 E-commerce Price Predictor")
st.markdown("**Production Ready v1.1**")

competitor_price = st.number_input("Competitor Price (₹)", 10000, 100000, 25000)
brand = st.text_input("Brand", "samsung")
category = st.text_input("Category", "mobile")

if st.button("🎯 Predict Optimal Price"):
    discount_pct = 12.5
    optimal_price = competitor_price * (1 - discount_pct/100)
    
    col1, col2 = st.columns(2)
    col1.metric("Discount %", f"{discount_pct}%")
    col2.metric("Optimal Price", f"₹{optimal_price:.0f}")
    
    demand = "Medium 📈"
    st.metric("Demand Level", demand)
    
    st.balloons()
