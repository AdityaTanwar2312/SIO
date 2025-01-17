import streamlit as st
from functions import *

# Page configuration
st.set_page_config(page_title="ESG Portfolio Optimizer", layout="wide")

# Title of the app
st.title("ESG Portfolio Optimizer")

# Layout with two columns
col1, col2 = st.columns([1, 1])  # Adjust ratios if needed

# Left Column: Inputs
with col1:
    st.header("Input Preferences")

    # ESG Scores
    environmental_score = st.slider("Environmental Score (0-100)", 0, 100, 50)
    social_score = st.slider("Social Score (0-100)", 0, 100, 50)
    governance_score = st.slider("Governance Score (0-100)", 0, 100, 50)
    # Risk Aversion
    risk_aversion = st.slider("Risk Aversion (0-5)", 0, 5, 3)
    esg_list = [environmental_score, social_score, governance_score, risk_aversion]

    # Excluded Sectors
    excluded_sectors = st.text_input("Excluded Sectors (comma-separated)", "")
    exstock_list = [excluded_sectors]

    # Excluded Stocks
    excluded_stocks = st.text_input("Excluded Stocks (comma-separated)", "")
    exsector_list = [excluded_stocks]

    # Process Button
    process_button = st.button("Process Portfolio")

# Right Column: Output
with col2:
    if process_button:
        st.header("Optimized Portfolio")
        with st.spinner("Generating Portfolio"):
            figre = get_portfolio(esg_list, exsector_list, exstock_list)
            st.pyplot(figre)
            st.success("Processing completed! Displaying your portfolio below.")
    else:
        st.info("Your optimized portfolio will be displayed here after processing.")