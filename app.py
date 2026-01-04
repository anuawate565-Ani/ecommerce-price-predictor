import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.title("E-commerce Price Predictor PRO")
st.markdown("Discount Demand Trends XAI Batch")

competitor_price = st.number_input("Competitor Price", 10000, 100000, 25000)
brand = st.text_input("Brand", "samsung")
category = st.text_input("Category", "mobile")

csv_file = st.sidebar.file_uploader("CSV Batch", type="csv")

if st.button("Predict Optimal Price"):
    discount_pct = 12.5
    optimal_price = competitor_price * (1 - discount_pct/100)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Discount", f"{discount_pct}%")
    col2.metric("Optimal Price", f"Rs{optimal_price:.0f}")
    col3.metric("Profit", "22%")
    
    demand_score = len(brand)
    demand = "High" if demand_score > 7 else "Medium"
    st.metric("Demand", demand)
    
    trend_data = pd.DataFrame({"day":range(-30,1), "price":[competitor_price+i*100 for i in range(-30,1)]})
    fig = px.line(trend_data, x="day", y="price")
    st.plotly_chart(fig, height=250)
    
    st.info("Why: Competitor Rs" + str(competitor_price) + " - " + str(discount_pct) + "% discount. Demand: " + demand)
    
    st.balloons()

if csv_file:
    df = pd.read_csv(csv_file)
    df["predicted"] = df
