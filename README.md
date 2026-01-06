# Student Performance Prediction: Binary Classification with Logistic Regression

**Predicting student pass/fail outcomes using behavioral and demographic features to enable early intervention strategies**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Technical Architecture](#technical-architecture)
- [Installation & Setup](#installation--setup)
- [Data Engineering Approach](#data-engineering-approach)
- [Model Development & Rationale](#model-development--rationale)
- [Results & Performance](#results--performance)
- [Key Insights & Business Value](#key-insights--business-value)
- [Future Improvements](#future-improvements)

---

## Project Overview

This project demonstrates my approach on end-to-end data science methodology by developing a **binary classification model** to predict whether students will pass (Grades A-C) or fail (Grades D-F) based on 12 behavioral and demographic features.

### Key Metrics
- **Accuracy**: 89%
- **Balanced Accuracy**: 87%
- **ROC-AUC**: 0.93
- **F1-Score**: 83%
- **Precision (Fail Detection)**: 84%

### Dataset
- **Source**: [Kaggle - Student Performance Data](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)
- **Size**: 2,392 student records
- **Features**: 12 predictors (demographics, study habits, parental involvement, extracurricular activities)
- **Target**: Binary pass/fail outcome (logically generated from th e5 tier grade classification)

---

## Problem Statement

### Challenge
Educational institutions often adopt **reactive** rather than **proactive** approaches to student support, resulting in interventions occurring only after academic performance via grades. This results in:

- Wasted tutoring resources on students unlikely to benefit
- Missed opportunities for early intervention with at-risk students to improve their chance at passing
- Inability to quantify which behavioral factors most influence academic outcomes

### Potential Solution
A **logistic regression classifier** that:
1. **Identifies at-risk students** before their assessments
2. **Quantifies feature importance** to guide targeted interventions (e.g., addressing absences vs. study time and which ones are most important)
3. **Provides probability scores of passing** (0-1)

### Impact
- **Early identification**: Flag at-risk students early on and not after big assessments likely contributing to their final grade
- **Resource optimisation**: Allocate priority tutoring to students with >70% predicted failure probability
- **Data-driven policy**: Evidence that reducing absences has more impact than increasing study time for example

---

## Technical Architecture

### Why Python?

Python is an indutsty standard tool with many packages to deal with data science and machine learning. Here are some of the main ones I used in this project:

**Pandas** (Data cleaning & manipulation) Industry-standard for tabular data; `DataFrame` structure intuitive for feature engineering |
**Scikit-learn** (Model training & evaluation) Amazing package to split into train and test datasets, while also assisting with model training and evaluating performance metrics.
**Statsmodels** (Statistical inference) Provides p-values & confidence intervals for hypothesis testing (not available in Scikit-learn) |
**Seaborn/Matplotlib** (Visualisation) Publication-quality plots; heatmaps ideal for correlation analysis |

### Model Choice: Logistic Regression

**Why not other algorithms?**

**Linear Regression** - Requires continuous target variable (This project uses binary pass/fail)
**Random Forest** - Black-box model which is not suitable. Stakeholders (educators/administrators in this case) require **interpretable** coefficients to justify targeted interventions
**Neural Networks** - Overkill for tabular data with 12 features; overfitting risk with only 2,392 samples

**Why Logistic Regression?**
1. **Interpretable coefficients**: Each feature's impact on failure risk is quantifiable (e.g., "1 additional absence increases failure odds by 12%")
2. **Probability outputs**: Returns probabiity not just binary predictions, enabling risk tiers such as low, medium, and high.
3. **Statistical rigor**: Compatible with hypothesis testing (p-values, confidence intervals using statsmodels)
4. **Computational efficiency**: Trains in seconds making it lightweight and applicable in various dashboards

### Trade-offs Acknowledged
- **Limitation**: Assumes linear relationship between features and log-odds of failure
- **Future Work**: Investigate a comparison with Gradient Boosting (XGBoost) for potential accuracy gains while maintaining feature importance explainability

---

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip
```

### Clone Repository
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
```

### Use pip intsall for the required packages like this:
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
pandas
numpy
scikit-learN
statsmodels
matplotlib
seaborn
jupyter
```

### Open and run the Notebook using the likes of vs code and jupyter

### Project Structure
```
dspp/
├── student_performance_analysis.ipynb  # Main analysis notebook
├── Student_performance_data _.csv      # Raw dataset
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
```

---

## Data Engineering Approach

### 1. Data Quality Assessment (Gov UK Framework)

Before any modeling, I conducted a **data quality audit** against the **UK Government Data Quality Framework**(https://www.gov.uk/government/publications/the-government-data-quality-framework/the-government-data-quality-framework#Data-quality-dimensions) to ensure trustworthy results.

#### Why This Framework?
- **Industry standard**
- **Thorough**: Covers 5 critical dimensions (vs. ad-hoc checks)

#### Results

| Dimension | Target | Achieved | Test Method |
|-----------|--------|----------|-------------|
| **Completeness** | 95%+ | 100% | `.isnull().sum()` across 16 features |
| **Accuracy** | No invalid ranges | 0 issues | Domain validation (GPA: 0-4.0, Age: 14-19, etc.) |
| **Validity** | Schema compliance | 0 issues | Binary fields contain only 0/1 |
| **Consistency** | Unique StudentIDs | 100% unique | No duplicate IDs via `.value_counts()` |
| **Uniqueness** | No duplicate rows | 100% unique records | `.duplicated().sum() == 0` |

**Outcome**: ** Perfect Data Quality** - Meaning no imputation or outlier treatment required & I could proceeded directly to feature engineering.

#### Code Example: Completeness Check
```python
# Check for missing values using dual methods
print("NULL values:\n", df.isnull().sum())
print("\nNA values:\n", df.isna().sum())

# Result: 0 missing values across all features
```

---

### 2. Feature Engineering: Target Variable Creation

**Challenge**: Original dataset contained 5-tier `GradeClass` (A/B/C/D/F). Binary classification requires two classes.

**Solution**: Engineered `PassFail` target column using pandas:

```python
# Pass = Grades A, B, C (GradeClass 0, 1, 2)
# Fail = Grades D, F (GradeClass 3, 4)
df['PassFail'] = (df['GradeClass'] < 3).astype(int)  # 1=Pass, 0=Fail
```

**Rationale**:
- Simplifies intervention logic (binary decision: provide support or not)
- Matches stakeholder language ("at-risk" vs. "high-performing")

**Class Distribution**:
- Pass: 32.1% (767 students)
- Fail: 67.9% (1,625 students)

**Imbalance Noted**: Used **stratified sampling** in train-test split to preserve the natural distribution from the sourced data.

<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/31a868fb-f069-4e3b-9f6f-e81445478170" />


---

### 3. Feature Selection: Preventing Target Leakage

**Dropped 3 columns** to avoid data leakage:

| Feature | Reason for Removal |
|---------|-------------------|
| `StudentID` | Non-predictive identifier (random assignment) |
| `GPA` | Grade Point Average is a **Big influence** for PassFail. It would achieve extremely high accuracy but unusable due to detecting at-risk students from features not directly related to the grade itself |
| `GradeClass` | Source of engineered target—retaining causes perfect multicollinearity |

**Retained 12 features to predict PassFail**:
```python
['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 
 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 
 'Sports', 'Music', 'Volunteering']
```

---

### 4. Train-Test Split Strategy

```python
df_train, df_test = train_test_split(
    student_df, 
    test_size=0.2,          # 80/20 split (industry standard for ML)
    random_state=1234,      # Reproducibility
    stratify=student_df['PassFail']  # Preserve class distribution with stratify
)
```

**Why Stratification?**
If I opted for random sampling, it could create a test set with 75% fails (vs. true 68%), leading to:
- Overly pessimistic accuracy estimates
- Biased threshold

**Verification**:
```
Train Set Distribution:    Test Set Distribution:
0 (Fail):  67.9%           0 (Fail):  67.9%
1 (Pass):  32.1%           1 (Pass):  32.1%
```

---

### 5. Feature Scaling: StandardScaler

**Applied**: Z-score normalisation (mean=0, std=1)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)  # Fits ONLY on training data

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Apply train parameters
```

**Why StandardScaler?**

| Consideration | StandardScaler | MinMaxScaler |
|---------------|----------------|--------------|
| **Outlier sensitivity** | Robust (uses standard deviation) | Sensitive (uses min/max) |
| **Feature distribution** | Works with any shape | Assumes bounded range |
| **Logistic regression compatibility** | Preferred | Can compress important signals |

**Critical Detail**: Fitted scaler **only on training data**, preventing **data leakage**. Test set scaled using training parameters simulates real-world deployment where future data statistics are unknown.

---

## Model Development & Rationale

### Exploratory Data Analysis (EDA)

#### Correlation Analysis
<img width="1020" height="777" alt="image" src="https://github.com/user-attachments/assets/53d98770-efff-4caa-ae7a-4f75103450d5" />


**Key Finding**: `Absences` shows strongest negative correlation with PassFail (-0.68), indicating:
- **1 additional absence** → 8% decrease in pass probability
- More impactful than `StudyTimeWeekly` (r=0.23)

**Business Implication**: Attendance monitoring systems should trigger alerts at **5 absences** (statistically significant threshold from logistic coefficients).

---

### Model Training

```python
from sklearn.linear_model import LogisticRegression

#initialise the model
model_scaled = LogisticRegression()

# train the model
model_scaled.fit(X_train_scaled, y_train)
```

**Hyperparameters**: Used defaults (no regularisation tuning) as:
1. High data quality (no noise to filter)
2. Only 12 features so there is a low overfitting risk
3. Baseline model prioritises interpretability

---

### Evaluation Metrics

#### Why Multiple Metrics?

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Accuracy** | Overall correctness | High-level performance (but misleading with class imbalance) |
| **Balanced Accuracy** | Average of sensitivity/specificity | **Corrects for 68% fail bias**, resulting in a true measure of model quality |
| **F1-Score** | Harmonic mean of precision/recall | Balances false positives vs. false negatives |
| **ROC-AUC** | Discrimination ability across thresholds | Measures separability of classes (0.5=random, 1.0=perfect) |

#### Results

```python
Accuracy:          89.1%
Balanced Accuracy: 87.39%  # Key metric for imbalanced data
F1-Score:          83.01%
ROC-AUC:           0.93   # Excellent discrimination (0.5 = random guessing, 1.0 = perfect discrimination)
```

**Interpretation**: 
- Model correctly classifies **87% of both pass AND fail students** (balanced accuracy)
- 93% probability that a randomly selected failing student scores higher risk than passing student (AUC)

---

### Confusion Matrix Analysis

<img width="780" height="624" alt="image" src="https://github.com/user-attachments/assets/045d4e92-c87c-4e71-abf3-91136b7fa39e" />


**Error Breakdown**:
- **False Positives (Type I)**: 5.22% (predicted fail but actually passed)
  - **Impact**: Unnecessary tutoring/support allocation (resource waste)
- **False Negatives (Type II)**: 5.64% (predicted pass but failed)
  - **Impact**: Missed intervention opportunities (**higher risk**)

**Model Behavior**: Slightly favors false negatives (5.64% vs 5.22%), could result from training on 68% fail distribution. 

**Acceptable for educational context where:**
- Slight under-identification of at-risk students is tolerable
- Low tolerance for "false alarms" that waste support resources

**Unacceptable for educational context where:**
- Missing an at-risk student has severe consequences (e.g., dropout, severe academic failure)
- Enough tutoring capacity exists to support more students
- Institutional priority is better to over-support than under-support

---

### ROC Curve
<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/8d832080-5a04-449c-8ff6-c9d37182c4b5" />


**AUC = 0.93** indicates model is **much better** than random guessing at distinguishing pass/fail students across all probability thresholds.

---

### Feature Importance (Coefficients)
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/f359e97a-551a-48d5-a159-d2a6b2e12187" />



| Rank | Feature | Coefficient | Interpretation | P-value |
|------|---------|-------------|----------------|---------|
| 1 | **Absences** | -2.7047 | Increased Absences = **HUGE negative effect** on passing (strongest predictor) | <0.001*** |
| 2 | **StudyTimeWeekly** | +0.5534 | Increased Study time = **Moderate positive effect** on passing | <0.001*** |
| 3 | **ParentalSupport** | +0.4688 | Increased Support level = **Moderate positive effect** on passing | <0.001*** |
| 4 | **Tutoring** | +0.3649 | Having tutor = **Moderate positive effect** on passing | <0.001*** |
| 5 | **Extracurricular** | +0.3430 | Participation = **Moderate positive effect** on passing | <0.001*** |

**Statistical Significance** (via statsmodels logistic regression):
- p<0.001 (Highly significant)
- p<0.01 (Very significant)
- p<0.05 (Significant)

---

## Key Insights & Business Value

**Actionable Insights**:
1. **Attendance is critical**: Absences have a **dramatically larger effect** than all other predictora combined (Around 5x stronger than the next predictor!)
2. **Study time matters**: Significant positive effect on outcomes
3. **Parental support helps**: Clear measurable benefit
4. **Tutoring shows improvement**: Meaningful improvement in passing likelihood
5. **Demographics show fairness**: Age (p=0.871), Gender (p=0.372), Ethnicity (p=0.852) are **not significant**, illustrating the model is unbiased and fair.

---

## Future Improvements

### Model Enhancements
- **Ensemble Methods**: Compare with Random Forest and XGBoost to evaluate accuracy gains while maintaining interpretability
- **Threshold Optimisation**: Adjust decision threshold (currently 0.5) to minimise false negatives in high-stakes educational contexts

### Feature Engineering
- **Interaction Terms**: Test `Absences × ParentalSupport` to identify if parental involvement improves attendance effects
- **Polynomial Features**: Investigate non-linear relationships such as diminishing returns of study time beyond 15 hours/week
