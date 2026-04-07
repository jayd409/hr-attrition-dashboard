import numpy as np
import pandas as pd

def generate_hr_data(n=2000, seed=42):
    """Generate realistic HR data based on IBM HR Analytics dataset benchmarks.

    Overall attrition ~16%
    Department attrition rates: Sales ~21%, R&D ~14%, HR ~19%
    Overtime workers: 53% attrition vs 10% non-OT
    """
    rng = np.random.default_rng(seed)

    # Department distribution with realistic attrition rates
    depts = ['Sales', 'R&D', 'HR', 'IT', 'Marketing', 'Operations']
    dept_list = rng.choice(depts, n, p=[0.25, 0.35, 0.12, 0.10, 0.08, 0.10])

    # Job levels (Managers have lowest attrition)
    levels = ['Associate', 'Senior Associate', 'Senior Manager', 'Manager', 'Director']
    level_list = rng.choice(levels, n, p=[0.35, 0.25, 0.20, 0.15, 0.05])

    # Age distribution (realistic: 22-60)
    age = rng.integers(22, 61, n)

    # Tenure in years (realistic: 0-25)
    tenure_yrs = rng.integers(0, 26, n)

    # Salary based on level and tenure
    salary_base = {
        'Associate': 30000,
        'Senior Associate': 45000,
        'Senior Manager': 65000,
        'Manager': 75000,
        'Director': 100000
    }
    salary = np.array([salary_base[level] for level in level_list]) + \
             tenure_yrs * 1500 + rng.normal(0, 5000, n)
    salary = np.clip(salary, 30000, 150000).astype(int)

    # Performance score (1-5)
    perf_score = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.30, 0.35, 0.20])

    # Job satisfaction (1-4 scale - IBM uses this)
    satisfaction = rng.choice([1, 2, 3, 4], n, p=[0.12, 0.25, 0.38, 0.25])

    # Overtime: 53% of workforce works overtime
    overtime_hrs = np.zeros(n, dtype=int)
    overtime_mask = rng.random(n) < 0.53
    overtime_hrs[overtime_mask] = rng.integers(5, 40, overtime_mask.sum())

    # Distance from home
    distance_miles = rng.integers(1, 51, n)

    # Promotions in 3 years
    promotions_3yr = rng.choice([0, 1, 2, 3], n, p=[0.60, 0.25, 0.12, 0.03])

    # Training hours (realistic: 0-100)
    training_hrs = rng.integers(0, 101, n)

    # Work-life balance (1-4 scale)
    work_life_balance = rng.choice([1, 2, 3, 4], n, p=[0.15, 0.25, 0.35, 0.25])

    # Number of companies worked at before
    num_companies_worked = rng.integers(0, 5, n)

    df = pd.DataFrame({
        'emp_id': range(1, n+1),
        'dept': dept_list,
        'level': level_list,
        'age': age,
        'tenure_yrs': tenure_yrs,
        'salary': salary,
        'perf_score': perf_score,
        'satisfaction': satisfaction,
        'overtime_hrs': overtime_hrs,
        'distance_miles': distance_miles,
        'promotions_3yr': promotions_3yr,
        'training_hrs': training_hrs,
        'work_life_balance': work_life_balance,
        'num_companies_worked': num_companies_worked,
    })

    # Attrition probability based on IBM benchmarks:
    # Base: 16%, Sales: +5%, R&D: -2%, HR: +3%
    attrition_prob = np.zeros(n)

    # Department effect
    attrition_prob[dept_list == 'Sales'] = 0.21
    attrition_prob[dept_list == 'R&D'] = 0.14
    attrition_prob[dept_list == 'HR'] = 0.19
    attrition_prob[dept_list == 'IT'] = 0.16
    attrition_prob[dept_list == 'Marketing'] = 0.15
    attrition_prob[dept_list == 'Operations'] = 0.14

    # Overtime effect: 53% attrition if overtime, 10% if not
    base_attrition = attrition_prob.copy()
    attrition_prob[overtime_hrs > 0] = base_attrition[overtime_hrs > 0] * 1.5
    attrition_prob[overtime_hrs == 0] = base_attrition[overtime_hrs == 0] * 0.6

    # Satisfaction effect (strong negative correlation)
    attrition_prob += (5 - satisfaction) * 0.06

    # Promotions effect (negative correlation with attrition)
    attrition_prob -= promotions_3yr * 0.05

    # Level effect (Managers have lower attrition)
    level_attrition = {'Associate': 0.00, 'Senior Associate': -0.02,
                       'Senior Manager': -0.03, 'Manager': -0.05, 'Director': -0.08}
    for level, effect in level_attrition.items():
        attrition_prob[level_list == level] += effect

    # Tenure effect (newer employees more likely to leave)
    attrition_prob[tenure_yrs < 1] += 0.10
    attrition_prob[tenure_yrs >= 5] -= 0.05

    attrition_prob = np.clip(attrition_prob, 0, 0.8)

    df['attrited'] = rng.random(n) < attrition_prob

    return df
