import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Page Config
st.set_page_config(page_title="AI Skewness Analyzer", layout="wide")

st.title("📊 Data Skewness Analyzer")
st.write("Apni CSV file upload karein aur numerical data ki skewness check karein.")

# File Uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Select Columns
    num_cols = df.select_dtypes(include='number').columns.tolist()
    
    if num_cols:
        target = st.selectbox("Column chunein:", num_cols)
        
        # Calculations
        df2 = df[[target]].copy()
        df2['Log_Transformed'] = np.log1p(df2[target])
        
        # Display Metrics
        c1, c2 = st.columns(2)
        c1.metric("Original Skew", f"{df2[target].skew():.2f}")
        c2.metric("Log Skew", f"{df2['Log_Transformed'].skew():.2f}")
        
        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df2[target], kde=True, color='red', ax=ax[0])
        ax[0].set_title("Before")
        sns.histplot(df2['Log_Transformed'], kde=True, color='green', ax=ax[1])
        ax[1].set_title("After Log")
        
        st.pyplot(fig)
    else:
        st.error("Is file mein koi numerical column nahi hai.")
else:
    st.info("Waiting for CSV file...")
