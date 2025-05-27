# 📊 Customer Churn Prediction

A complete end-to-end project to analyze and visualize customer churn behavior using **SQL**, **Python**, and **Power BI**.

---

## 🚀 Project Overview

This project focuses on understanding customer churn patterns and building data visualizations from a telecom dataset. The process includes:

- Exploratory Data Analysis using SQL (SQLite)
- Executing queries programmatically in Python
- Visualizing data insights in Power BI

---

## 🧱 Project Structure

```
customer-churn-prediction/
│
├── data/                     # Original data files
├── images/                   # Exported charts and dashboards
├── models/                   # (Optional) Future model training
├── notebooks/
│   └── 01_initial_analysis.ipynb  # Python notebook for SQL + Pandas analysis
├── output/                   # Query results exported as CSV
│   ├── avg_monthly_charge.csv
│   └── churn_by_contract.csv
├── sql/
│   ├── churn_project.db      # SQLite database
│   ├── create_tables.sql     # Table creation queries
│   └── queries.sql           # All SQL analysis queries
├── .gitignore
└── README.md                 # You are here 🚀
```

---

## 🛠️ Tools & Technologies

- **SQLite** – for storing and querying data  
- **Python (pandas + sqlite3)** – for running queries and exporting results  
- **Power BI** – for dashboard and data storytelling  
- **VS Code + Git** – for development and version control  

---

## 📌 Key Insights

- 📈 **Average Monthly Charge**: 64.76  
- 🔍 Churn Rate is significantly higher for **Month-to-Month** contracts  
- ✅ Final dashboards created and shared in Power BI

---

## 🔗 How Everything is Connected

| Step         | Tool        | Description |
|--------------|-------------|-------------|
| Data Import  | SQLite      | Raw data imported into a `.db` file |
| Analysis     | SQL         | Queries written in DB Browser & run in Python |
| Export       | Python      | Results saved as `.csv` using `pandas` |
| Dashboard    | Power BI    | CSVs loaded for visual analysis |

---

## 📎 Example SQL Query

```sql
SELECT AVG(MonthlyCharges) AS avg_monthly_charge FROM telco_data;
```

## 📎 Example Python Snippet

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("sql/churn_project.db")
query = "SELECT * FROM telco_data"
df = pd.read_sql(query, conn)
```

---

## 📊 Power BI

- Imported both raw and exported CSVs
- Created KPI cards, bar charts, and interactive filters

---

## ✅ Status

- [x] Data Prepared
- [x] SQL Queries Executed
- [x] Python Notebook Complete
- [x] Power BI Dashboard Built
- [ ] Model Training (coming soon)

---

## 🙋‍♀️ Author

**(Pkhoei)**  
Exploring data science, machine learning, and data visualization.