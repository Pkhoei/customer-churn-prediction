# 📉 Customer Churn Prediction

This project analyzes customer churn behavior in a telecom dataset and explores the factors associated with customer attrition using Python, SQL, and data visualization. The next phase will extend the analysis toward machine learning–based churn prediction.


---


## 🚀 Project Objectives

- Explore customer characteristics and churn patterns through EDA.
- Identify key factors associated with customer churn.
- Analyze customer behavior using Python and SQL.
- Visualize churn patterns and business-relevant insights.
- Develop and evaluate machine learning models for churn prediction in the next phase.

---

## 🧱 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── telco_customer_churn.csv     # Original dataset
│
├── notebooks/
│   └── 01_initial_analysis.ipynb    # Initial exploratory analysis
│
├── .gitignore
└── README.md
```

---

## 🛠️ Tools & Technologies

- **Python** – data cleaning, exploratory analysis, and preprocessing
- **pandas** – data manipulation and analysis
- **Matplotlib / Seaborn** – data visualization
- **scikit-learn** – machine learning modeling and evaluation (next phase)
- **Jupyter Notebook** – analysis and experimentation
- **Git + GitHub** – version control and project documentation

---
## Exploratory Data Analysis

After data cleaning, the dataset contains **7,032 customer records**.

The target variable shows a moderate class imbalance: **73.4%** of customers were retained, while **26.6%** churned.

This imbalance should be considered during model evaluation, particularly when selecting metrics such as **precision, recall, F1-score, and ROC-AUC**.
![Customer Churn Distribution](images/churn_distribution.png)

---

## 📊 Key Insights

- 📈 **Average Monthly Charge:** 64.76  
- 🔍 **Churn Rate:** Much higher in **Month-to-Month** contract customers  

---


## ✅ Project Status

- [x] Dataset added
- [x] Initial exploratory analysis
- [ ] Data cleaning & preprocessing
- [ ] Extended exploratory data analysis
- [ ] Machine learning modeling
- [ ] Model evaluation


