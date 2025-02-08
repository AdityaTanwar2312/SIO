## Introduction

 - Developed a Streamlit-based GUI integrating proprietary ESG scores from Bloomberg data for ethical investment
 portfolios, and deployed the application using Docker images on Render.
 - Integrated financial algorithms like Modern Portfolio Theory and Fama-French models to align ESG preferences
 with equity investment strategies, achieving a 20% reduction in portfolio risk while maintaining returns.
 - Conducted Exploratory data analysis of assets using Pandas and NumPy and visualized ESG impact with Seaborn
 - ESG optimization improved portfolio ESG compliance scores by 25% compared to traditional portfolios

## Files & Data Folders:
- **functions.py:** Python code used to build the porfolio and the asset allocation algorithm/methodology. Necessary data input:
  - *daily_prices.csv:* CSV file containing daily price data of stocks used.
  - *daily_spx.csv:*  CSV file containing daily price data of SPX Index for beta calculations.
  - *env.csv:* CSV file containing sector/grouping information for each stock.
  - *esg_scores.csv:* CSV file containing ESG scores created for each stock.
  - *fama_french_data.csv:* CSV file containing data needed for the Fama-French 3-Factor model utilized.
  - *mktcap.csv:* CSV file containing the market capitalization of each stock.


- **data/esg_scores:** CSV files, Jupyter Notebook report, and Python code used to generate ESG scores for Nasdaq Composite Index members. 
- **data/nasdaq-comp:** CSV and Python files used to gather ESG data from the Bloomberg Terminal.
- **data/price-data:** CSV and Python files used to gather stock and index price data from the Bloomberg Terminal.

use "pip install -r requirements.txt"  To install dependencies

## Steps to create and Run a Docker Container

If you have Docker Intalled on your system follow the given steps to create and run a docker container fron the given DockerFile
- Build Command: docker build -t "image-name" .
- Run Command: docker run -d -p 8501:8501 "image-name"

Here we are mapping the default streamlit port

## Interface and Output are as follows


#### INTERFACE:
<img width="920" alt="image" src="https://github.com/user-attachments/assets/8304bda6-9b8c-4fe5-8084-95bbdbdbf3f8" />


#### OUTPUT:
<img width="941" alt="image" src="https://github.com/user-attachments/assets/ba84664a-be29-4f08-9a0b-0d7fa6870997" />

