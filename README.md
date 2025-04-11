# 🛒 Predicting Online Purchasers' Intention

This project focuses on building a machine learning model to predict whether an online shopper will complete a purchase based on their browsing behavior. The model uses real-world behavioral data from an e-commerce platform to support conversion rate optimization and customer engagement strategies.

---

## 📁 Project Structure
```
purchase-intention-prediction/
│
├── data/                     # Raw and processed dataset files
│   └── online_shoppers_intention.csv
│
├── notebooks/               # Jupyter notebooks used for EDA, modeling, evaluation
│   ├── 01_data_cleaning.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
│
├── models/                  # Saved models (.pkl or .joblib)
│   └── xgboost_model.pkl
│
├── scripts/                 # Python scripts for modular pipelines
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── outputs/                 # Visualizations, plots, reports
│   └── feature_importance.png
│
├── README.md                # Project overview and documentation
├── requirements.txt         # List of dependencies
└── LICENSE                  # License info
```

---

## 📁 Dataset
**Name:** Online Shoppers Purchasing Intention Dataset  
**Source:** UCI Machine Learning Repository  
**Link:** [https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

**Size:** 12,330 sessions  
**Target Variable:** `Revenue` (True = purchase, False = no purchase)

**Feature Examples:**
- Administrative_Duration
- Informational_Duration
- ProductRelated_Duration
- BounceRates
- ExitRates
- PageValues
- VisitorType, Month, TrafficType, Region, etc.

---

## 🧠 Project Goal
To develop a high-performing and interpretable predictive model that classifies user sessions based on their likelihood to result in a purchase, enabling real-time decision-making for e-commerce applications.

---

## 🔧 Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn, PyCaret
- XGBoost, LightGBM, Random Forest
- SMOTE (for class balancing)
- Jupyter Notebook

---

## 🔍 Methodology
1. **Data Preprocessing:**
   - One-hot encoding of categorical variables
   - Feature scaling and cleaning
   - Class balancing using SMOTE

2. **Model Training:**
   - Applied ensemble methods including XGBoost, LightGBM, and Random Forest
   - Used Grid Search for hyperparameter optimization

3. **Model Evaluation:**
   - Accuracy: 93.54% (XGBoost)
   - Metrics used: Precision, Recall, F1-score, AUC

4. **Feature Importance:**
   - `PageValues` and `ProductRelated_Duration` were the most influential predictors

---

## 📊 Results
- **Best Model:** XGBoost
- **Top Features:** PageValues, ProductRelated_Duration, BounceRates
- **Use Case:** Real-time intent scoring, personalized discounting, checkout optimization

---

## 📚 Academic Component
- Structured Literature Review (Tables 7.1–7.7)
- APA-style citations for 15+ peer-reviewed studies
- Key methods discussed: Technology Acceptance Model (TAM), Ensemble Learning, Clickstream Analysis

---

## ✅ Endgame / Future Scope
- Deployable scoring API for intent prediction
- Integration with live e-commerce platforms
- Real-time interventions: offers, retargeting, chat prompts

---

## 👥 Team & Acknowledgments
**Lead Analyst:** [Navya Nanduri]  
**Associate Analyst:** [Yatheesh Nagella]

---

## 📎 License
This repository is intended for academic and research use only.
