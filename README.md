# 📱 Mobile Behaviour Analysis — Data Analytics Project

> **DAUP (Data Analytics & Understanding Project)**  
> A complete end-to-end data analytics project analyzing mobile usage behaviour of 1,000 users using Python, statistical analysis, and machine learning models.

---

## 📌 Project Overview

This project explores how people use their mobile phones — how much time they spend on screen, which types of apps they use (social media, productivity, gaming), and whether their usage patterns can be predicted or classified using machine learning.

The goal is to apply real-world data analytics techniques — from exploratory data analysis (EDA) to supervised machine learning — on a structured mobile behaviour dataset.

---

## 🎯 Objectives

- Understand the distribution and structure of mobile usage data
- Perform Univariate, Bivariate, and Multivariate exploratory analysis
- Detect outliers and check for missing/duplicate data
- Apply **Simple** and **Multiple Linear Regression** to predict app usage
- Build a **Logistic Regression** model to classify users as High or Low app users
- Evaluate models using MSE, MAE, R², Accuracy, Confusion Matrix, and Classification Report
- Diagnose **Overfitting** and **Underfitting** by comparing Train vs Test performance

---

## 📂 Project Structure

```
mobile-behaviour-analysis/
│
├── mobile-behavioral.csv          # Dataset (1000 users, 10 features)
├── project.ipynb                  # Main Jupyter Notebook (full project code)
├── README.md                      # Project documentation (this file)
└── LICENSE                        # License — All Rights Reserved
```

---

## 📊 Dataset

| Property        | Details                                      |
|-----------------|----------------------------------------------|
| Source          | Kaggle — Mobile Behaviour Dataset            |
| Rows            | 1,000 users                                  |
| Columns         | 10 features                                  |
| Missing Values  | 0                                            |
| Duplicates      | 0                                            |

### Columns Description

| Column                        | Type        | Description                                      |
|-------------------------------|-------------|--------------------------------------------------|
| `User_ID`                     | Discrete    | Unique identifier for each user                  |
| `Age`                         | Discrete    | User age (range: 18 – 59 years)                  |
| `Gender`                      | Nominal     | Male (517) / Female (483)                        |
| `Total_App_Usage_Hours`       | Continuous  | Total daily app usage in hours (mean: 6.41 hrs)  |
| `Daily_Screen_Time_Hours`     | Continuous  | Total daily screen time in hours (mean: 7.70 hrs)|
| `Number_of_Apps_Used`         | Discrete    | Number of apps used per day (mean: 16.65)        |
| `Social_Media_Usage_Hours`    | Continuous  | Time on social media apps (mean: 2.46 hrs)       |
| `Productivity_App_Usage_Hours`| Continuous  | Time on productivity apps (mean: 2.50 hrs)       |
| `Gaming_App_Usage_Hours`      | Continuous  | Time on gaming apps (mean: 2.48 hrs)             |
| `Location`                    | Nominal     | City — New York, Chicago, Houston, Phoenix, LA   |

---

## 🛠️ Technologies & Libraries

```python
import pandas as pd                    # Data loading and manipulation
import numpy as np                     # Numerical operations
import matplotlib.pyplot as plt        # Base plotting

import seaborn as sns                  # Statistical visualizations

from sklearn.model_selection import train_test_split          # Data splitting
from sklearn.linear_model import LinearRegression             # Regression model
from sklearn.linear_model import LogisticRegression           # Classification model

from sklearn.metrics import mean_squared_error                # MSE metric
from sklearn.metrics import mean_absolute_error               # MAE metric
from sklearn.metrics import r2_score                          # R² metric
from sklearn.metrics import accuracy_score                    # Accuracy metric
from sklearn.metrics import confusion_matrix                  # Confusion matrix
from sklearn.metrics import classification_report             # Full report
```

---

## 🔍 Project Workflow

```
1. Data Loading          →  pd.read_csv()
2. Data Understanding    →  .head(), .info(), .describe(), .shape
3. Data Type Analysis    →  Discrete, Continuous, Nominal classification
4. EDA — Univariate      →  Histograms, Boxplots (screen time, gaming, apps)
5. EDA — Bivariate       →  Scatter plots (apps vs usage, screen time vs social media)
6. EDA — Multivariate    →  Heatmap, Pairplot, Catplot
7. Missing & Outliers    →  .isnull(), .duplicated(), IQR Boxplot method
8. Spread Analysis       →  Mean, Median, Std Dev, Skewness, Kurtosis
9. Regression (Scatter)  →  Correlation, Covariance, Regression line
10. Train-Val-Test Split  →  60% Train / 20% Validation / 20% Test
11. Simple Regression     →  1 Feature → Predict Total App Usage
12. Multiple Regression   →  2 Features → Predict Total App Usage
13. Overfitting Check     →  Compare Train R² vs Test R²
14. Classification        →  Binary target (High=1 / Low=0), Logistic Regression
15. Model Evaluation      →  MSE, MAE, R², Accuracy, Confusion Matrix, F1
```

---

## 📈 Key Results

### Regression Models

| Model               | Train R²   | Validation R² | Test R²    | Status       |
|---------------------|------------|---------------|------------|--------------|
| Simple Linear Reg.  | ≈ 0.00018  | ≈ −0.0028     | ≈ 0.0013   | Underfitting |
| Multiple Linear Reg.| ≈ 0.0013   | ≈ 0.0012      | ≈ 0.0020   | Underfitting |

> **MSE ≈ 9.66 | MAE ≈ 2.68 hrs | R² ≈ 0.0013**

### Classification Model

| Metric    | Value     |
|-----------|-----------|
| Accuracy  | ≈ 49.5%   |
| Precision | ≈ 0.48–0.52 |
| Recall    | ≈ 0.46–0.53 |
| F1 Score  | ≈ 0.49–0.50 |

---

## 🔑 Key Findings

- **Screen time averages 7–8 hours/day** and is consistent across all age groups, genders, and cities — no major demographic differences detected.
- **App usage is evenly split**: Social Media ≈ 33.1% | Productivity ≈ 33.6% | Gaming ≈ 33.3% — no single category dominates.
- **All distributions are nearly normal** — skewness ranges from −0.11 to +0.04, indicating balanced data with no significant skew.
- **All models underfit** — R² ≈ 0 and Accuracy ≈ 50% confirm that daily screen time and number of apps used are weak predictors of total usage.
- **No overfitting was found** — training scores were also near zero, confirming underfitting rather than memorization.
- **Future improvement**: Feature engineering, additional variables, and advanced algorithms (Random Forest, SVM, Neural Networks) are recommended for stronger predictive performance.

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/mobile-behaviour-analysis.git
cd mobile-behaviour-analysis
```

**2. Install required libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**3. Open the notebook**
```bash
jupyter notebook project.ipynb
```

**4. Run all cells**  
Go to `Kernel → Restart & Run All`

---

## 📋 Requirements

```
python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 📚 Concepts Covered

| Concept                  | Description                                                   |
|--------------------------|---------------------------------------------------------------|
| Data Types               | Quantitative (Discrete, Continuous) & Qualitative (Nominal)  |
| EDA                      | Univariate, Bivariate, Multivariate analysis                 |
| Outlier Detection        | IQR method via Boxplots                                       |
| Spread & Distribution    | Mean, Median, Std Dev, Skewness, Kurtosis                    |
| Correlation              | Heatmap, Covariance, Pearson Correlation                     |
| Regression Analysis      | Simple & Multiple Linear Regression                           |
| Supervised Learning      | Train / Validation / Test split (60-20-20)                   |
| Model Evaluation         | MSE, MAE, R² Score                                           |
| Classification           | Binary classification with Logistic Regression               |
| Confusion Matrix         | TP, TN, FP, FN analysis                                      |
| Overfitting/Underfitting | Diagnosis by comparing Train vs Test scores                  |

---

## 🏫 Academic Information

- **Course:** Data Analytics & Understanding Project (DAUP)
- **Domain:** Mobile Behaviour Analytics
- **Dataset Source:** Kaggle
- **Supervised by:** Prof. Arti J. Patel

---

## ⚠️ License

This project is protected under a **Custom Restrictive License**.

```
Copyright (c) 2025 — All Rights Reserved

This project, including all code, notebooks, analysis, and documentation,
is the sole intellectual property of the original author.

Permissions:
  ✅  You MAY view and read this project for learning purposes.
  ✅  You MAY reference this project with proper credit.

Restrictions:
  ❌  You MAY NOT copy, reproduce, or reuse any part of this code.
  ❌  You MAY NOT submit this project as your own academic work.
  ❌  You MAY NOT distribute, publish, or upload this project elsewhere.
  ❌  You MAY NOT use this project commercially without written permission.

Unauthorized use, reproduction, or academic submission of this project
is strictly prohibited and may result in academic and/or legal consequences.

See the LICENSE file for full terms.
```

---

## 📬 Contact

For any queries or collaborations, please open an **Issue** on this repository.

---

<p align="center">
  Made with | DAUP(Data Analytics Using Pyton) Project | Mobile Behaviour Analysis
</p>
