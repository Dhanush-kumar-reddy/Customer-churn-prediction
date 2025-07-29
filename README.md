# 🚀 Customer Churn Prediction

A complete end-to-end data science project that predicts customer churn using a machine learning model and deploys it as an interactive web application.

---

### 🔗 Live Demo

**[Click here to view the live application](customer-churn-prediction-msyazgdcnhfjal25rmfq9t.streamlit.app)**

---

### 🎯 Problem Statement

The goal of this project is to build a classification model that accurately predicts whether a customer of a telecom company is likely to churn (cancel their service). By identifying at-risk customers, the company can offer targeted incentives to improve retention and reduce revenue loss.

---

### 🛠️ Tech Stack

- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** Scikit-learn, XGBoost
- **Web App:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git, GitHub, Git LFS

---

### 📈 Results

After cleaning the data, performing feature engineering, and tuning multiple models (Logistic Regression, Random Forest, XGBoost), the final model selected was **Logistic Regression**.

- **Final Test Set Accuracy:** 80%
- **Final Test Set F1-Score (for Churn):** 0.60

This demonstrates that a well-tuned, simpler model can sometimes outperform more complex ones for certain datasets.

---

### 📖 Project Summary

This project covers the complete data science lifecycle:
1.  **Data Cleaning & EDA:** Handled missing values and visualized key patterns affecting churn.
2.  **Feature Engineering & Preprocessing:** Used one-hot encoding and created new features to improve model performance.
3.  **Model Training & Tuning:** Systematically trained and compared multiple models using `GridSearchCV` to find the best hyperparameters.
4.  **Deployment:** Built and deployed an interactive Streamlit web app for real-time predictions.

---