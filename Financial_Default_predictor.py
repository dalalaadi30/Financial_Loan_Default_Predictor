#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv("data/cs-training.csv")

df.info()


# In[2]:


df.head(50)


# In[3]:


df['SeriousDlqin2yrs'].value_counts()


# In[4]:


df.drop(columns=["Unnamed: 0"], inplace = True)
df.head()


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='SeriousDlqin2yrs', data=df)
plt.title("Loan Default Distribution")


# In[6]:


df.hist(bins=30, figsize=(15, 12))
plt.tight_layout()


# In[7]:


sns.boxplot(x=df['MonthlyIncome'])
sns.boxplot(x=df['DebtRatio'])


# In[8]:


corr = df.corr()
plt.figure(figsize=(9, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")


# In[9]:


sns.boxplot(x='SeriousDlqin2yrs', y='MonthlyIncome', data=df)


# In[10]:


sns.histplot(data=df, x='age', hue='SeriousDlqin2yrs', bins=30, kde=True)


# In[11]:


#sns.histplot(df['RevolvingUtilizationOfUnsecuredLines'], bins=100)
#plt.xlim(0, 2)  # clip for readability


# In[12]:


late_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
]

for col in late_cols:
    sns.countplot(x=col, hue='SeriousDlqin2yrs', data=df)
    plt.title(f'{col} vs Default')
    plt.show()


# In[13]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, r2_score
import joblib


# In[14]:


# 🎯 Step 5: Classification - Loan Default Prediction
X_cls = df.drop('SeriousDlqin2yrs', axis=1)
y_cls = df['SeriousDlqin2yrs'] 

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

rf_cls = RandomForestClassifier()
rf_cls.fit(X_train_cls, y_train_cls)
y_pred_cls = rf_cls.predict(X_test_cls)

print("Classification Report:\n", classification_report(y_test_cls, y_pred_cls))
print("ROC AUC:", roc_auc_score(y_test_cls, rf_cls.predict_proba(X_test_cls)[:, 1]))



# In[15]:


#joblib.dump(rf_cls, "default_model.pkl")


# In[16]:


# 📊 Step 6: Regression - Credit Score Forecasting
# Create a synthetic credit score (300-850 scale)
df['custom_credit_score'] = 850 \
    - df['RevolvingUtilizationOfUnsecuredLines'] * 100 \
    - df['NumberOfTime30-59DaysPastDueNotWorse'] * 20 \
    - df['NumberOfTime60-89DaysPastDueNotWorse'] * 25 \
    - df['NumberOfTimes90DaysLate'] * 30 \
    - df['DebtRatio'] * 50

df['custom_credit_score'] = df['custom_credit_score'].clip(lower=300, upper=850)

df.head()


# In[17]:


X_reg = df.drop(['SeriousDlqin2yrs', 'custom_credit_score'], axis=1)
y_reg = df['custom_credit_score']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor()
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(rf_reg, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, random_state=42, scoring='neg_mean_squared_error')
search.fit(X_train_reg, y_train_reg)

best_rf_reg = search.best_estimator_
y_pred_reg = best_rf_reg.predict(X_test_reg)

print("R2 Score:", r2_score(y_test_reg, y_pred_reg))


# In[18]:


print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))


# In[19]:


#joblib.dump(best_rf_reg, "score_model.pkl")


# In[20]:


import matplotlib.pyplot as plt

feat_importances = pd.Series(best_rf_reg.feature_importances_, index=X_reg.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()


# In[21]:


test_df = pd.read_csv("data/cs-test.csv")
test_df.drop(columns=["Unnamed: 0","SeriousDlqin2yrs"], inplace=True)
test_df.head()


# In[22]:


test_df.fillna(test_df.median(), inplace=True)
test_df.head(10)


# In[23]:


# Load the trained model
import joblib
default_model = joblib.load("default_classifier.pkl")

# Predict
default_preds = default_model.predict(test_df)

# Optional: Probabilities
default_probs = default_model.predict_proba(test_df)[:, 1]

# Save results
submission_df = pd.DataFrame({
    "Id": test_df.index,
    "PredictedDefault": default_preds,
    "DefaultProbability": default_probs
})


# In[24]:


submission_df.head(50)


# In[25]:


score_model = joblib.load("score_model.pkl")

credit_score_preds = score_model.predict(test_df)

# Save results
credit_score_df = pd.DataFrame({
    "Id": test_df.index,
    "PredictedCreditScore": credit_score_preds
})


# In[26]:


credit_score_df.head(10)


# In[27]:


final_results = pd.DataFrame({
    "Id": test_df.index,
    "PredictedDefault": default_preds,
    "DefaultProbability": default_probs,
    "PredictedCreditScore": credit_score_preds
})


# In[28]:


final_results.head(10)


# In[29]:


final_results.to_csv("final_financial_advice.csv", index=False)


# In[30]:


sns.histplot(final_results["PredictedCreditScore"], bins=30)
plt.title("Predicted Credit Score Distribution")
plt.show()

sns.countplot(x="PredictedDefault", data=final_results)
plt.title("Predicted Default Counts")
plt.show()


# In[31]:


sns.scatterplot(
    x=final_results['PredictedCreditScore'],
    y=final_results['DefaultProbability'],
    hue=final_results['PredictedDefault'],
    palette='coolwarm',
    alpha=0.6
)
plt.title("Credit Score vs Default Probability")
plt.xlabel("Predicted Credit Score")
plt.ylabel("Default Probability")
plt.legend(title="Predicted Default")
plt.grid(True)
plt.show()


# In[32]:


# Categorize based on credit score
def risk_bucket(score):
    if score >= 750:
        return "Low Risk"
    elif score >= 600:
        return "Moderate Risk"
    else:
        return "High Risk"

final_results["RiskCategory"] = final_results["PredictedCreditScore"].apply(risk_bucket)

# Countplot
sns.countplot(x="RiskCategory", data=final_results, palette="Set2")
plt.title("Customer Distribution by Risk Category")
plt.show()


# In[33]:


default_counts = final_results['PredictedDefault'].value_counts()
labels = ['No Default', 'Default']
plt.pie(default_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff6666'])
plt.axis('equal')
plt.title("Predicted Default Breakdown")
plt.show()


# In[34]:


sns.boxplot(x="RiskCategory", y="DefaultProbability", data=final_results, palette="pastel")
plt.title("Default Probability by Risk Category")
plt.show()


# In[35]:


sns.kdeplot(final_results["PredictedCreditScore"], fill=True, color='purple')
plt.title("Predicted Credit Score Density")
plt.xlabel("Score")
plt.ylabel("Density")
plt.show()


# In[ ]:




