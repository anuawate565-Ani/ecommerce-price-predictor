import streamlit as st
import numpy as np
st.title("🛒 E-commerce Price Predictor")
competitor_price = st.number_input("Competitor Price (₹)", 10000, 100000, 25000)
brand = st.text_input("Brand", "samsung")
if st.button("🎯 Predict Price"):
 brand_factor = len(brand)
 category_factor = len(brand)
 optimal_price = competitor_price * 0.95 - (brand_factor + category_factor) * 50
 st.success(f"**Price: ₹{optimal_price:.0f}**")
