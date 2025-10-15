# app.py - Enhanced Streamlit UI for Financial Advisor ML Project

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ------------------- Page Configuration -------------------
st.set_page_config(page_title="💰 Financial Advisor App", layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #f8f9fa;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stButton > button {
            background-color: #0066cc;
            color: white;
            border-radius: 8px;
            padding: 8px 20px;
        }
        .stDownloadButton > button {
            background-color: #28a745;
            color: white;
            border-radius: 8px;
        }
        .risk-low {
            background-color: #d4edda;
            color: #155724;
            padding: 5px;
            border-radius: 5px;
        }
        .risk-moderate {
            background-color: #fff3cd;
            color: #856404;
            padding: 5px;
            border-radius: 5px;
        }
        .risk-high {
            background-color: #f8d7da;
            color: #721c24;
            padding: 5px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- Title and Description -------------------
st.title("💰 Financial Advisor - Default & Credit Score Prediction")

st.markdown("""
Welcome to the **Financial Advisor App**! This tool helps predict:
- Whether a person is likely to default on a loan
- An estimate of their credit score

👈 Upload your CSV in the sidebar to begin, or review the sample demo below to understand the model.
""")

# ------------------- Load and Show Demo Visualizations -------------------
@st.cache_data
def load_example():
    return pd.read_csv("data/sample_input.csv")

example_df = load_example()
example_df.fillna(example_df.median(numeric_only=True), inplace=True)
example_df = example_df[(example_df['age'] > 0) & (example_df['RevolvingUtilizationOfUnsecuredLines'] <= 2)]

st.subheader("📊 Sample Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demo Monthly Income Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(example_df['MonthlyIncome'], kde=True, bins=30, color='royalblue', ax=ax)
    st.pyplot(fig)

with col2:
    st.markdown("**Demo Age Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(example_df['age'], kde=True, bins=30, color='darkcyan', ax=ax)
    st.pyplot(fig)

# ------------------- Upload CSV -------------------
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV file", type=["csv"])

threshold_low = st.sidebar.slider("Low Risk Threshold", min_value=700, max_value=800, value=750)
threshold_moderate = st.sidebar.slider("Moderate Risk Threshold", min_value=500, max_value=699, value=600)

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    test_df.drop(columns=["Unnamed: 0"], errors='ignore', inplace=True)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(test_df.head(), use_container_width=True)

    test_df.fillna(test_df.median(numeric_only=True), inplace=True)
    test_df = test_df[(test_df['age'] > 0) & (test_df['RevolvingUtilizationOfUnsecuredLines'] <= 2)]

    @st.cache_resource
    def load_models():
        return joblib.load("default_classifier.pkl"), joblib.load("score_model.pkl")

    clf_model, reg_model = load_models()

    st.subheader("🔮 Running Predictions")
    features_for_models = test_df.copy()
    test_df['PredictedDefault'] = clf_model.predict(features_for_models)
    test_df['DefaultProbability'] = clf_model.predict_proba(features_for_models)[:, 1]
    test_df['PredictedCreditScore'] = reg_model.predict(features_for_models)

    st.success("✅ Predictions complete!")

    st.subheader("📈 Predicted Results")
    st.dataframe(test_df[['PredictedDefault', 'DefaultProbability', 'PredictedCreditScore']].head(), use_container_width=True)

    def risk_bucket(score):
        if score >= threshold_low:
            return "Low Risk"
        elif score >= threshold_moderate:
            return "Moderate Risk"
        else:
            return "High Risk"

    test_df["RiskCategory"] = test_df["PredictedCreditScore"].apply(risk_bucket)

    # ------------------- Visual Insights -------------------
    st.subheader("📊 Visual Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📌 Credit Score Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(test_df['PredictedCreditScore'], kde=True, bins=30, color='slateblue', ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("**📌 Default Probability by Risk Category**")
        fig, ax = plt.subplots()
        sns.boxplot(x='RiskCategory', y='DefaultProbability', data=test_df, palette='Set2', ax=ax)
        st.pyplot(fig)

    st.markdown("**📌 Credit Score vs Default Probability**")
    fig, ax = plt.subplots()
    sns.scatterplot(x='PredictedCreditScore', y='DefaultProbability', hue='RiskCategory', data=test_df, alpha=0.7, ax=ax)
    st.pyplot(fig)

    st.markdown("**📌 Default Prediction Breakdown**")
    pie_data = test_df['PredictedDefault'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=['No Default', 'Default'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff6666'])
    ax.axis('equal')
    st.pyplot(fig)

    # ------------------- Feature Importance -------------------
    st.subheader("🧮 Top Contributing Features")

    if hasattr(clf_model, 'coef_'):
        importances = pd.Series(clf_model.coef_[0], index=features_for_models.columns)
    elif hasattr(clf_model, 'feature_importances_'):
        importances = pd.Series(clf_model.feature_importances_, index=features_for_models.columns)
    else:
        importances = pd.Series(dtype='float64')

    top_features = importances.abs().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis", ax=ax)
    ax.set_title("Top Features Affecting Default Prediction")
    st.pyplot(fig)

    # ------------------- Correlation Heatmap -------------------
    st.subheader("📌 Feature Correlation Heatmap")
    corr = test_df[features_for_models.columns].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ------------------- Summary -------------------
    st.subheader("📌 Prediction Summary")
    st.markdown(f"**Total Records Analyzed:** {len(test_df)}")
    st.markdown(f"**Default Rate:** {test_df['PredictedDefault'].mean():.2%}")
    st.markdown(f"**Average Credit Score:** {test_df['PredictedCreditScore'].mean():.2f}")
    st.markdown(f"**High Risk Customers:** {(test_df['RiskCategory'] == 'High Risk').sum()}")

    # ------------------- High Risk Table -------------------
    st.subheader("🚨 Top High Risk Customers")
    st.dataframe(test_df.sort_values("DefaultProbability", ascending=False).head(5), use_container_width=True)

    # ------------------- Download Section -------------------
    st.subheader("💾 Download Final Results")
    output_df = test_df[["PredictedDefault", "DefaultProbability", "PredictedCreditScore", "RiskCategory"]]
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=output_df.to_csv(index=False).encode('utf-8'),
        file_name="financial_predictions.csv",
        mime="text/csv"
    )

    # ------------------- Model Info -------------------
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        - **Default Prediction**: Logistic Regression
        - **Credit Score Estimation**: Linear Regression
        - **Features used**: Revolving utilization, age, late payments, debt ratio, monthly income, etc.
        - **Dataset Source**: Kaggle 'Give Me Some Credit'
        """)

else:
    st.info("☝️ Upload your CSV file from the sidebar to begin analysis.")
