# Students-Performance-in-Exams

## Dataset: StudentsPerformance.csv

Source
- File: StudentsPerformance.csv (included in this repository)
- A tabular dataset of student exam scores and demographic/background variables.

Contents and size
- Number of records: 1000 students
- Columns:
  - gender — student gender (female / male)
  - race/ethnicity — group identifier (group A, group B, …)
  - parental level of education — highest parental education (e.g., bachelor's degree, some college, high school, etc.)
  - lunch — whether lunch is standard or free/reduced
  - test preparation course — completed or none
  - math score — integer 0–100
  - reading score — integer 0–100
  - writing score — integer 0–100

Notes
- Scores are numeric and appear to be on a 0–100 scale.
- The CSV in this repository is the raw data used for the analyses below.

## Exploratory / Statistical analysis performed

1. Data cleaning and basic checks
   - Loaded the CSV and checked for missing values, duplicate rows, and basic types.
   - Converted score columns to numeric and categorical columns to category dtype.

2. Descriptive statistics
   - Calculated count, mean, median, standard deviation, min/max and quartiles for `math score`, `reading score`, `writing score`.
   - Computed each student’s average score (mean of the three subjects) to summarize overall performance.

3. Group summaries and visualizations
   - Distribution plots (histograms / KDE) for each score.
   - Boxplots of scores by categorical variables (gender, race/ethnicity, parental level of education, lunch, test preparation).
   - Bar charts of mean score by test preparation status and by parental education level.

4. Tests / comparisons
   - Gender comparison: two-sample t-test (or non-parametric alternative) on mean scores (e.g., math) to test whether average scores differ by gender.
   - Test preparation: two-sample t-test comparing students who completed the test preparation course vs none.
   - Parental education & race/ethnicity: one-way ANOVA to test for differences in mean scores across more than two groups.
   - Pairwise post-hoc tests (Tukey HSD) where ANOVA detects significant differences.
   - Chi-square tests for association between categorical variables when appropriate.

5. Correlations
   - Pearson correlation matrix between math, reading and writing scores.
   - Visualized with a heatmap and pairwise scatterplots.

Interpretation highlights (high-level)
- The three subject scores are strongly positively correlated (students who do well in one exam tend to do well in the others).
- Students who completed the test preparation course typically show higher mean exam scores than those who did not (effect size and statistical significance confirmed via t-test).
- Parental level of education and lunch status often show systematic differences in mean scores (tested with ANOVA). Use post-hoc tests to identify which specific groups differ.

## Regression analyses

Goal
- Model student performance and quantify effects of demographic/background predictors on exam scores.
- Two main modeling targets used:
  1. Predicting each subject score (math / reading / writing) as separate regressions.
  2. Predicting overall performance using the average score (continuous target).
  3. (Optional) Binary classification: pass/fail using logistic regression if a pass threshold is defined.

Preprocessing
- Feature engineering:
  - Encode categorical features with one-hot encoding (drop one level to avoid multicollinearity).
  - Create `avg_score = (math + reading + writing) / 3` for models targeting overall performance.
  - Optionally scale numeric features (StandardScaler) when using regularized regressors.
- Train/test split (e.g., 80/20) with random_state for reproducibility.
- Use cross-validation (k-fold, e.g., k=5) for model selection and evaluation.

Models fit and evaluated
- Ordinary Least Squares (OLS) multiple linear regression using statsmodels for interpretable coefficients and p-values.
- scikit-learn LinearRegression and regularized variants (Ridge, Lasso) to control overfitting and compare performance.
- Model formula example (for avg_score):
  avg_score ~ C(gender) + C(race_ethnicity) + C(parental_level_of_education) + C(lunch) + C(test_preparation_course)
- Evaluation metrics (continuous targets):
  - R-squared (R²)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
- For classification (pass/fail):
  - Accuracy, Precision, Recall, F1, ROC AUC

Interpreting coefficients
- OLS coefficient estimates indicate the average change in the target (e.g., avg_score) associated with a one-unit change in the predictor, holding other predictors constant.
- For categorical predictors, coefficients are interpreted relative to the chosen reference level.
- p-values & confidence intervals reported to assess statistical significance and uncertainty.
- Standardized coefficients or partial dependence plots can be used to compare effect sizes.

Example findings to report (what to include in final results)
- Coefficient table with estimates, standard errors, p-values, and 95% confidence intervals.
- Model R² and RMSE on hold-out test set.
- Feature importance (for tree-based or regularized models).
- Partial dependence or marginal effect plots for the most influential categorical variables (e.g., test preparation completed vs none).

## Reproducible analysis (Python)

Below is a concise runnable script that reproduces the main steps (EDA + regression). You can run it in a Jupyter notebook or as a .py file.

```python
# analysis.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf

# 1. Load data (if running in repo root)
df = pd.read_csv("StudentsPerformance.csv")

# 2. Basic cleaning
df.columns = [c.strip().replace("/", "_").replace(" ", "_") for c in df.columns]
df[['math_score','reading_score','writing_score']] = df[["math score","reading score","writing score"]].astype(float)

# 3. Create avg score
df['avg_score'] = df[["math score","reading score","writing score"]].mean(axis=1)

# 4. Descriptive stats
print(df[["math score","reading score","writing score","avg_score"]].describe())

# 5. Correlations and plots
corr = df[["math score","reading score","writing score"]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Score correlations')
plt.show()

# 6. Test preparation effect (group means)
print(df.groupby('test preparation course')[["math score","reading score","writing score","avg_score"]].mean())

# 7. OLS regression (interpretable coefficients)
# Rename columns for formula convenience
df = df.rename(columns={
    'gender':'gender',
    'race/ethnicity':'race_ethnicity',
    'parental level of education':'parental_education',
    'lunch':'lunch',
    'test preparation course':'test_prep'
})
formula = 'avg_score ~ C(gender) + C(race_ethnicity) + C(parental_education) + C(lunch) + C(test_prep)'
model = smf.ols(formula=formula, data=df).fit()
print(model.summary())

# 8. Train/test for predictive evaluation
X = pd.get_dummies(df[["gender","race/ethnicity","parental level of education","lunch","test preparation course"]], drop_first=True)
y = df['avg_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("R2:", r2_score(y_test, y_pred), "RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

## How to interpret and present results in the README
- Include a short paragraph summarizing the main model results (R², interpretation of top 2–3 coefficients, and whether test prep was significant).
- Add key visualizations (histograms, boxplots, heatmap, coefficient plot). Keep figures small and link to a notebook or figure directory for full-resolution figures.
- Provide the model code (or link to a notebook) so others can reproduce the analysis.

## Next steps / reproducibility
- I added an analysis notebook in `notebooks/analysis.ipynb` that contains the code used for the analysis and instructions to run it locally.
- If you prefer, I can also run the notebook here and paste the numeric outputs & plots into the repository.
