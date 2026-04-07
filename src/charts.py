import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from utils import save_html
from analysis import attrition_by_dept, overtime_impact

def chart_attrition_by_dept(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    dept_stats = attrition_by_dept(df)
    colors = ['red' if x > 0.2 else 'orange' if x > 0.15 else 'green'
              for x in dept_stats['attrition_rate']]
    ax.barh(dept_stats['dept'], dept_stats['attrition_rate'] * 100, color=colors, alpha=0.7)
    ax.set_xlabel('Attrition Rate (%)')
    ax.set_title('Attrition Rate by Department')
    ax.set_xlim(0, max(dept_stats['attrition_rate'] * 100) * 1.1)
    return fig

def chart_feature_importance(feature_imp):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_imp.index, feature_imp.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Coefficient Magnitude')
    ax.set_title('Feature Importance for Attrition Prediction')
    return fig

def chart_satisfaction_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    satisfaction_bins = df.groupby('satisfaction')['attrited'].agg(['sum', 'count'])
    satisfaction_bins['rate'] = satisfaction_bins['sum'] / satisfaction_bins['count']
    ax.scatter(satisfaction_bins.index, satisfaction_bins['rate'] * 100, s=200, alpha=0.6, color='teal')
    ax.plot(satisfaction_bins.index, satisfaction_bins['rate'] * 100, color='teal', linestyle='--', alpha=0.5)
    ax.set_xlabel('Satisfaction Score (1-5)')
    ax.set_ylabel('Attrition Rate (%)')
    ax.set_title('Satisfaction vs Attrition Rate')
    ax.set_xticks(range(1, 6))
    ax.grid(True, alpha=0.3)
    return fig

def chart_age_distribution(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    attrited = df[df['attrited']]['age']
    retained = df[~df['attrited']]['age']
    ax.hist([retained, attrited], bins=15, label=['Retained', 'Attrited'],
           color=['green', 'red'], alpha=0.6)
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.set_title('Age Distribution (Attrited vs Retained)')
    ax.legend()
    return fig

def chart_overtime_impact(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    ot_impact = overtime_impact(df)
    bins_labels = [str(x) for x in ot_impact.index]
    ax.bar(range(len(ot_impact)), ot_impact.values * 100, color='coral', alpha=0.7)
    ax.set_xticks(range(len(ot_impact)))
    ax.set_xticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    ax.set_ylabel('Attrition Rate (%)')
    ax.set_title('Overtime Hours vs Attrition Rate')
    return fig

def chart_salary_by_level(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    levels = sorted(df['level'].unique())
    data_by_level = [df[df['level'] == l]['salary'].values for l in levels]
    bp = ax.boxplot(data_by_level, labels=levels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Salary ($)')
    ax.set_title('Salary Distribution by Level')
    ax.grid(True, alpha=0.3, axis='y')
    return fig

def build_dashboard(df, y_true, y_pred, feature_imp):
    charts = [
        ('Attrition Rate by Department', chart_attrition_by_dept(df)),
        ('Feature Importance', chart_feature_importance(feature_imp)),
        ('Satisfaction vs Attrition', chart_satisfaction_scatter(df)),
        ('Age Distribution', chart_age_distribution(df)),
        ('Overtime Hours Impact', chart_overtime_impact(df)),
        ('Salary Distribution by Level', chart_salary_by_level(df)),
    ]

    attrition_rate = df['attrited'].mean()
    y_binary = (y_pred >= 0.5).astype(int)
    model_acc = np.mean(y_true == y_binary)
    top_driver = feature_imp.index[0]

    kpis = [
        ('Attrition Rate', f"{attrition_rate*100:.1f}%"),
        ('Model Accuracy', f"{model_acc*100:.1f}%"),
        ('Top Driver', top_driver),
        ('Total Employees', len(df)),
    ]

    save_html(charts, 'HR Attrition Dashboard', kpis,
             'outputs/hr_dashboard.html')
