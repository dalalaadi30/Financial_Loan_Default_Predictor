# 💰 Financial Default Predictor

An end-to-end **Machine Learning + Streamlit** project that predicts:
- Whether an individual is likely to **default on a loan**
- Their **estimated credit score**

This project was developed using the **Give Me Some Credit (Kaggle)** dataset as part of our **B.Tech Final Year Project (2025, UIET MDU Rohtak)**.

---

## 🚀 Features
- **Loan Default Prediction** (Logistic Regression / Random Forest Classifier)
- **Credit Score Estimation** (Linear Regression / Random Forest Regressor)
- **Interactive Streamlit Dashboard**
- Risk Categorization: 🔴 High | 🟡 Moderate | 🟢 Low
- Visual Analytics: Histograms, Boxplots, Scatterplots, Feature Importances, Heatmaps
- Downloadable **CSV file of predictions**
- Clean and responsive UI with custom styling

---

## 🛠 Tech Stack
- **Python**: pandas, numpy, scikit-learn, joblib  
- **Visualization**: matplotlib, seaborn  
- **Deployment/UI**: Streamlit  
- **Experimentation**: Jupyter Notebook  

---
# Clone the repo
git clone https://github.com/<your-username>/Financial-Default-Predictor.git
cd Financial-Default-Predictor

# Create environment (conda or pip)
conda env create -f environment.yml
conda activate financial-default

# Download models (this will pull from Google Drive)
python download_models.py

# Run the app
streamlit run app/app.py
