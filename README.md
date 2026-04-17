# 📡 Telco Customer Churn Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-red?logo=scikit-learn&logoColor=white)


> **Predicting customer churn for a telecom company using exploratory data analysis and machine learning — enabling proactive retention strategies.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Methodology](#-methodology)
- [Key Findings & Insights](#-key-findings--insights)
- [Model Performance](#-model-performance)
- [How to Run](#-how-to-run)
- [Business Recommendations](#-business-recommendations)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🔍 Overview

Customer churn — when a customer stops doing business with a company — is a critical challenge for telecom providers. Acquiring a new customer costs **5–25× more** than retaining an existing one. This project builds a full end-to-end churn prediction pipeline: from raw data exploration to a deployable classification model that flags at-risk customers before they leave.

---

## 💼 Business Problem

A telecommunications company wants to reduce customer attrition. The goal is to:

1. **Identify** which customers are likely to churn.
2. **Understand** the key drivers of churn.
3. **Enable** the business to intervene proactively with targeted retention offers.

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | [IBM Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target** | `Churn` (Yes / No) |

### Feature Categories

| Category | Features |
|----------|---------|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## 🗂️ Project Structure

```
Telco-Churn-Analysis/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│
├── notebooks/
│   └── Telco_Churn_Analysis.ipynb              # Main analysis notebook
│
├── images/
│   ├── churn_distribution.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
├── requirements.txt                            # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| **Python 3.8+** | Core programming language |
| **Pandas & NumPy** | Data manipulation and numerical computing |
| **Matplotlib & Seaborn** | Data visualization |
| **Scikit-Learn** | Machine learning models and evaluation |
| **XGBoost** | Gradient boosting classifier |
| **Jupyter Notebook** | Interactive analysis environment |
| **Imbalanced-learn** | Handling class imbalance (SMOTE) |

---

## 🔬 Methodology

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

```
1. Business Understanding  →  Define churn prediction objective
2. Data Understanding      →  Explore distributions, correlations, missing values
3. Data Preparation        →  Clean, encode, scale, handle class imbalance
4. Modelling               →  Train & tune multiple classifiers
5. Evaluation              →  Compare models using AUC-ROC, F1-Score, Recall
6. Deployment (Insights)   →  Actionable retention recommendations
```

### Step-by-Step Process

#### 1. 🔎 Exploratory Data Analysis (EDA)
- Analyzed churn rate (~26.5% churn vs 73.5% retained)
- Visualized feature distributions using histograms, count plots, and box plots
- Investigated correlations between tenure, contract type, and churn
- Identified that **month-to-month contracts**, **high monthly charges**, and **lack of add-on services** are strong churn indicators

#### 2. 🧹 Data Preprocessing
- Converted `TotalCharges` from object to numeric; handled 11 missing values
- One-hot encoded categorical features
- Standardized numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`)
- Applied **SMOTE** to address the class imbalance (26% / 74% split)

#### 3. 🤖 Model Training
Trained and compared the following classifiers:

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble, handles feature interactions |
| XGBoost | Best-performing gradient boosting model |

Hyperparameter tuning performed using **GridSearchCV** with 5-fold cross-validation.

#### 4. 📈 Model Evaluation
Primary metric: **Recall** (minimise false negatives — churners predicted as retained) and **AUC-ROC**.

---

## 💡 Key Findings & Insights

| Finding | Impact |
|---------|--------|
| **Month-to-month contracts** have ~42% churn rate vs ~11% for one-year and ~3% for two-year contracts | Contract type is the single strongest predictor |
| Customers with **no OnlineSecurity or TechSupport** churn at nearly **2× the rate** of customers with these services | Add-on services significantly boost retention |
| **Fibre optic** internet customers churn more than DSL customers, possibly due to higher prices | Pricing and perceived value drive dissatisfaction |
| Customers in the **first 12 months (low tenure)** are at the highest churn risk | Early engagement programs are critical |
| **Electronic check** payment users churn more than those using automatic payments | Automatic payment adoption correlates with commitment |
| **Senior citizens** have a ~41% churn rate vs ~24% for non-seniors | Targeted senior retention programs needed |

---

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.1% | 65.4% | 55.8% | 60.2% | 0.843 |
| Decision Tree | 78.6% | 58.7% | 57.2% | 57.9% | 0.762 |
| Random Forest | 81.3% | 68.0% | 57.5% | 62.3% | 0.858 |
| **XGBoost** | **82.4%** | **69.2%** | **60.1%** | **64.3%** | **0.872** |

> ✅ **Best Model: XGBoost** — highest AUC-ROC and balanced precision/recall, making it suitable for real-world deployment.

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Anu-gautam/Telco-Churn-Analysis.git
cd Telco-Churn-Analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook notebooks/Telco_Churn_Analysis.ipynb
```

### Requirements (`requirements.txt`)

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
xgboost>=1.4.0
imbalanced-learn>=0.8.0
jupyter>=1.0.0
```

---

## 📌 Business Recommendations

Based on the analysis, the following retention strategies are recommended:

1. **🎯 Offer long-term contract incentives** — Customers on month-to-month plans churn at the highest rate. Offer discounts to migrate them to annual contracts.
2. **🔒 Bundle security & support services** — Customers without `OnlineSecurity` or `TechSupport` churn significantly more. Bundle these as free trials.
3. **🆕 Strengthen new-customer onboarding** — Target customers in their first year with loyalty rewards and dedicated support.
4. **💳 Encourage automatic payments** — Electronic check users are higher-risk. Incentivise auto-pay enrollment.
5. **👴 Create senior citizen programs** — Higher churn rates among seniors suggest a need for simplified plans and dedicated support channels.
6. **📉 Review fibre optic pricing** — Despite being a premium service, fibre optic customers churn more — investigate price-value perception.

---

## 🔮 Future Work

- [ ] Deploy the model as a REST API using **FastAPI** or **Flask**
- [ ] Build an interactive dashboard with **Streamlit** or **Power BI**
- [ ] Incorporate **SHAP values** for explainable AI (model interpretability)
- [ ] Experiment with **LightGBM** and **Neural Networks**
- [ ] Create a **customer lifetime value (CLV)** model to prioritise high-value churners
- [ ] Automate retraining with new data using **MLflow**

---

## 👩‍💻 Author

**Anu Gautam**

[![GitHub](https://img.shields.io/badge/GitHub-Anu--gautam-black?logo=github)](https://github.com/Anu-gautam)

---

*If you found this project useful, please consider giving it a ⭐ on GitHub!*
