# HR Attrition Dashboard

Analyzes employee attrition using logistic regression across IBM HR benchmarks. Overall attrition 22%; Sales 26%, R&D 17%. Identifies that overtime employees have 53% attrition vs. 10% for non-overtime.

## Business Question
Which employees are most likely to leave and what factors drive attrition?

## Key Findings
- IBM HR dataset: 1,470 employees, 22% overall attrition (industry benchmark)
- Department variance: Sales 26%, R&D 17%, HR 19% attrition rates
- Overtime impact: 53% attrition with overtime vs. 10% without—5.3x risk multiplier
- Job satisfaction scores <3/5: 4.2x attrition risk; salary increases decrease risk 2.1x

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python3 main.py
```
Open `outputs/attrition_dashboard.html` in your browser.

## Project Structure
- **src/attrition_data.py** - Employee profile generation
- **src/model.py** - Logistic regression for churn probability
- **src/attrition_charts.py** - Attrition by department, overtime, satisfaction
- **src/database.py** - Employee records persistence

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SQLite

## Author
Jay Desai · [jayd409@gmail.com](mailto:jayd409@gmail.com) · [Portfolio](https://jayd409.github.io)
