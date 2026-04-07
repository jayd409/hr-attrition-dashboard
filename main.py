#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import numpy as np
from attrition_data import generate_hr_data
from model import train, predict, feature_importance, accuracy
from attrition_charts import build_dashboard
from database import save_to_db, query

df = generate_hr_data(n=2000)
save_to_db(df, 'employees')

# Prepare features for logistic regression
feature_cols = ['age', 'tenure_yrs', 'salary', 'perf_score', 'satisfaction',
                'overtime_hrs', 'distance_miles', 'promotions_3yr', 'training_hrs']
X = df[feature_cols].values
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)  # Normalize
y = df['attrited'].astype(int).values

# Train model
theta = train(X, y, lr=0.05, epochs=400)
y_pred = predict(X, theta)
model_acc = accuracy(y, y_pred)

# Get feature importance
feat_imp = feature_importance(theta, feature_cols)

attrition_rate = df['attrited'].mean()
top_driver = feat_imp.index[0]

print(f"Attrition Rate: {attrition_rate*100:.1f}%")
print(f"Model Accuracy: {model_acc*100:.1f}%")
print(f"Top driver: {top_driver}")

build_dashboard(df, y, y_pred, feat_imp)

print("\n--- SQL Analytics (SQLite) ---")
attrition_by_dept = query("""
    SELECT dept,
           COUNT(*) as total_employees,
           SUM(CASE WHEN attrited = 1 THEN 1 ELSE 0 END) as attrited_count,
           ROUND(100.0 * SUM(CASE WHEN attrited = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as attrition_rate
    FROM employees
    GROUP BY dept
    ORDER BY attrition_rate DESC
""")
print("\nAttrition Rate by Department:")
print(attrition_by_dept.to_string(index=False))

attrition_by_level = query("""
    SELECT level,
           COUNT(*) as total_employees,
           SUM(CASE WHEN attrited = 1 THEN 1 ELSE 0 END) as attrited_count,
           ROUND(100.0 * SUM(CASE WHEN attrited = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as attrition_rate
    FROM employees
    GROUP BY level
    ORDER BY attrition_rate DESC
""")
print("\nAttrition Rate by Level:")
print(attrition_by_level.to_string(index=False))

salary_by_attrition = query("""
    SELECT CASE WHEN attrited = 1 THEN 'Attrited' ELSE 'Retained' END as status,
           COUNT(*) as count,
           ROUND(AVG(salary), 0) as avg_salary,
           ROUND(MIN(salary), 0) as min_salary,
           ROUND(MAX(salary), 0) as max_salary
    FROM employees
    GROUP BY attrited
""")
print("\nAverage Salary by Attrition Status:")
print(salary_by_attrition.to_string(index=False))

print("Dashboard complete!")
