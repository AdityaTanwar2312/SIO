import streamlit as st
from functions import *

# Page configuration
st.set_page_config(page_title="ESG Portfolio Optimizer", layout="wide")

# Title of the app
st.title("ESG Portfolio Optimizer")

# Layout with two columns
col1, col2 = st.columns([1, 1])  # Adjust ratios if needed

stock_options = ['AAPL', 'AMZN', 'TSLA', 'MSFT', 'NVDA', 'ADBE', 'JPM', 'BAC', 'C', 'PFE', 'JNJ', 'UNH', 'BA', 'GE', 'XOM', 'CVX', 'DUK', 'NEE', 'KO', 'PG', 'PEP', 'T', 'VZ', 'META', 'GOOGL', 'NFLX', 'NEM', 'CAT', 'DHI', 'PLD', 'SPG']
sector_options = ['Consumer Discretionary', 'Information Technology', 'Financials', 'Health Care', 'Industrials', 'Energy', 'Utilities', 'Consumer Staples', 'Communication Services', 'Materials', 'Real Estate']

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

    # Excluded Stocks
    excluded_stocks = st.multiselect(
        "Select stocks to exclude:",
        options=stock_options,
        default=None,  # Default is no selection
    )

    # Excluded Sectors
    excluded_sectors = st.multiselect(
        "Select sectors to exclude:",
        options=sector_options,
        default=None,  # Default is no selection
    )

    # Process Button
    process_button = st.button("Process Portfolio")

# Right Column: Output
with col2:
    if process_button:
        st.header("Optimized Portfolio")
        with st.spinner("Generating Portfolio"):
            figre = get_portfolio(esg_list, excluded_sectors, excluded_stocks)
            st.pyplot(figre)
            st.success("Processing completed! Displaying your portfolio below.")
    else:
        st.info("Your optimized portfolio will be displayed here after processing.")