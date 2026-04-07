import pandas as pd
import numpy as np

def attrition_by_dept(df):
    g = df.groupby('dept').agg(
        count=('emp_id', 'count'),
        attrited=('attrited', 'sum')
    ).reset_index()
    g['attrition_rate'] = (g['attrited'] / g['count']).round(3)
    return g.sort_values('attrition_rate', ascending=False)

def attrition_by_level(df):
    g = df.groupby('level').agg(
        count=('emp_id', 'count'),
        attrited=('attrited', 'sum')
    ).reset_index()
    g['attrition_rate'] = (g['attrited'] / g['count']).round(3)
    return g

def attrition_by_age_band(df):
    df = df.copy()
    df['age_band'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100],
                            labels=['<30', '30-40', '40-50', '50+'])
    g = df.groupby('age_band').agg(
        count=('emp_id', 'count'),
        attrited=('attrited', 'sum')
    ).reset_index()
    g['attrition_rate'] = (g['attrited'] / g['count']).round(3)
    return g

def avg_satisfaction_by_dept(df):
    return df.groupby('dept')['satisfaction'].mean().round(2)

def salary_gap_analysis(df):
    attrited_avg = df[df['attrited']]['salary'].mean()
    retained_avg = df[~df['attrited']]['salary'].mean()
    return {'attrited_avg': round(attrited_avg, 2), 'retained_avg': round(retained_avg, 2)}

def overtime_impact(df):
    """Correlation between overtime and attrition."""
    df = df.copy()
    df['overtime_bin'] = pd.cut(df['overtime_hrs'], bins=5)
    return df.groupby('overtime_bin', observed=True)['attrited'].mean()
