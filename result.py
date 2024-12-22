import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
results_raw = pd.read_csv("results_df_raw.csv")  # Model performance on raw data
results_preprocessed = pd.read_csv("result_df.csv")  # Model performance on preprocessed data
raw_data = pd.read_csv("raw_data.csv")  # Raw dataset
preprocessed_data = pd.read_csv("processed_dataset.csv")  # Preprocessed dataset

# Streamlit layout
st.title("Model Performance Dashboard")
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Data Distribution", "Model Performance"])

# Section 1: Data Distribution
if section == "Data Distribution":
    st.header("Data Distribution")
    
    # Select feature for distribution visualization
    selected_feature = st.selectbox("Select a feature to visualize", raw_data.columns)
    
    # Distribution of raw data
    st.subheader("Raw Data Distribution")
    fig_raw = px.histogram(raw_data, x=selected_feature, title=f"{selected_feature} Distribution (Raw Data)")
    st.plotly_chart(fig_raw)
    
    # Distribution of preprocessed data
    st.subheader("Preprocessed Data Distribution")
    fig_preprocessed = px.histogram(preprocessed_data, x=selected_feature, title=f"{selected_feature} Distribution (Preprocessed Data)")
    st.plotly_chart(fig_preprocessed)
    
    st.write("Observations:")
    st.write("""
    - Raw data may contain missing values, duplicates, or outliers, which can distort the distribution.
    - Preprocessed data shows cleaner distributions due to scaling, imputation, and other preprocessing steps.
    """)

# Section 2: Model Performance
elif section == "Model Performance":
    st.header("Model Performance Comparison")
    
    # Combine results for comparison
    results_combined = pd.concat(
        [results_raw.assign(Dataset="Raw Data"), results_preprocessed.assign(Dataset="Preprocessed Data")],
        axis=0
    )
    
    # Visualization: Accuracy comparison
    st.subheader("Model Accuracy Comparison")
    fig_accuracy = px.bar(
        results_combined,
        x="Model",
        y="Accuracy",
        color="Dataset",
        barmode="group",
        title="Accuracy Comparison Between Raw and Preprocessed Data"
    )
    st.plotly_chart(fig_accuracy)
    
    st.write("Observations:")
    st.write("""
    - Preprocessed data typically improves accuracy by removing noise and inconsistencies.
    - Tree-based models like Random Forest may show marginal improvement, while linear models like Logistic Regression benefit significantly.
    """)

