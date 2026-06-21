# 📊 Superstore Profit Prediction using Machine Learning

## 🧠 Overview
This project applies data analysis and machine learning techniques to predict **Profit** based on **Sales** and **Discount** using a retail dataset.

The goal is to understand key business drivers of profitability and build predictive models using Python.

---

## 📂 Dataset
The dataset contains retail transaction data including:
- Sales
- Discount
- Profit
- Category, Region, City, etc.

---

## 🎯 Objective
To build a regression model that predicts Profit and evaluates which factors influence it most.

---

## 🔧 Workflow

### 1. Data Cleaning
- Converted Order Date to datetime format
- Extracted month and year features
- Checked missing values and duplicates

### 2. Exploratory Data Analysis (EDA)
- Correlation analysis
- Sales and profit trends by category, region, city, month, and year
- Data visualization using Seaborn and Matplotlib

### 3. Machine Learning Models
Two regression models were built:

- Linear Regression (baseline model)
- Random Forest Regressor (advanced model)

---

## 📊 Model Evaluation

| Model               | MAE    | RMSE   | R² Score |
|--------------------|--------|--------|----------|
| Linear Regression  | 158.78 | 196.18 | 0.36     |
| Random Forest      | 168.35 | 215.82 | 0.22     |

### Key Insight:
Linear Regression performed better, indicating that the relationship between features and profit is mostly linear.

---

## 📈 Key Findings
- Sales is the most important predictor of Profit
- Discount has a weaker influence on Profit
- Profit trends vary across months, categories, and regions
- Business data shows mostly linear relationships

---

## 🧠 Skills Demonstrated
- Python (Pandas, NumPy)
- Data Cleaning & Feature Engineering
- Data Visualization (Matplotlib, Seaborn)
- Machine Learning (Regression Models)
- Model Evaluation (MAE, RMSE, R²)
- Business Insight Interpretation

---

## 📌 Conclusion
This project demonstrates an end-to-end machine learning workflow from data exploration to predictive modeling and business insight extraction.

It highlights practical skills in applying machine learning to real-world retail data.
