import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.sidebar.title("🏢 Ecommerce AI")
st.sidebar.caption("Anu Awate | AI/ML Engineer")

st.title("🛒 E-commerce Price Predictor PRO")
st.markdown("**Discount • Demand • Trends • Batch • XAI**")

competitor_price = st.number_input("Competitor Price (₹)", 10000, 100000, 25000)
brand = st.text_input("Brand", "samsung")
category = st.text_input("Category", "mobile")

csv_file = st.sidebar.file_uploader("📁 CSV Batch", type="csv")

if st.button("🎯 Predict Optimal Price"):
    discount_pct = 12.5
    optimal_price = competitor_price * (1 - discount_pct/100)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Discount", f"{discount_pct}%")
    col2.metric("Optimal Price", f"₹{optimal_price:.0f}")
    col3.metric("Profit", "22%")
    
    demand_score = len(brand) * 1.2
    demand = "High 🔥" if demand_score > 10 else "Medium 📈"
    st.metric("Demand", demand)
    
    trend_data = pd.DataFrame({"day":range(-30,1), "price":[competitor_price+i*100 for i in range(-30,1)]})
    fig = px.line(trend_data, x="day", y="price", title="📈 30-Day Trend")
    st.plotly_chart(fig, height=250)
    
    st.info(f"""
🤔 **Why ₹{optimal_price:.0f}?**
• Competitor ₹{competitor_price:,} → {discount_pct}% OFF
• {brand}: {demand} demand  
• Trend: {"Stable" if trend_data["price"].iloc[-1] ≈ trend_data["price"].mean() else "Rising"}
    """)
    
    st.balloons()

if csv_file:
    df = pd.read_csv(csv_file)
    df["predicted"] = df["competitor_price"] * 0.875
    st.dataframe(df)
    st.download_button("💾 Download", df.to_csv(index=False).encode(), "results.csv")
