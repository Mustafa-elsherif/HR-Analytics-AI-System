# HR Analytics AI System
## An AI-Powered System for Employee Analytics and Decision Support

---

## Project Overview

This project develops an end-to-end AI-powered HR analytics system using Apache Spark and Machine Learning to analyze employee data, predict attrition, evaluate performance, segment employees, and recommend targeted HR strategies.

The system processes the IBM HR Analytics Dataset containing 1,470 employees and 35 features, delivering actionable insights to support data-driven HR decision-making.

---

## Team Members

| Name | Student ID | Responsibilities |
|------|-----------|-----------------|
| Mustafa Nabil | 240103415 | Model Development, Clustering, Dashboard |
| Mahmoud Hesham | 240101375 | Model Development, Clustering, Dashboard |
| Mariam Magdy | 240102374 | Exploratory Data Analysis (EDA) |
| Omar Ahmed Wafik | 240101244 | Data Preprocessing, Feature Engineering |
| Mohamed Nour | 240100780 | Data Ingestion, Insights and HR Recommendations |

---

## Project Structure

HR-Analytics-AI-System/

├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   ├── hr_data.parquet
│   ├── hr_data_preprocessed.parquet
│   └── hr_data_engineered.parquet

├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_development.ipynb
│   ├── 06_clustering.ipynb
│   └── 07_insights.ipynb

├── src/
│   └── dashboard.py

├── models/
│   ├── attrition_model/
│   └── performance_model/

├── outputs/
│   ├── attrition_distribution.png
│   ├── attrition_by_department.png
│   ├── attrition_by_gender.png
│   ├── attrition_by_age.png
│   ├── income_vs_performance.png
│   ├── salary_by_department.png
│   ├── overtime_vs_attrition.png
│   ├── correlation_heatmap.png
│   ├── feature_engineering.png
│   ├── feature_importance.png
│   ├── elbow_method.png
│   ├── cluster_visualization.png
│   └── high_risk_analysis.png

├── requirements.txt
├── packages.txt
└── README.md

---

## Dataset

| Property | Details |
|----------|---------|
| Source | IBM HR Analytics Employee Attrition Dataset (Kaggle) |
| Records | 1,470 employees |
| Features | 35 columns |
| Target Variable | Attrition (Yes / No) |

---

## Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.13 | Main programming language |
| Apache Spark | 3.5.8 | Distributed data processing |
| PySpark | 3.5.8 | Spark Python API |
| Scikit-learn | Latest | Machine learning |
| Pandas | Latest | Data manipulation |
| NumPy | Latest | Numerical computing |
| Matplotlib | Latest | Visualization |
| Seaborn | Latest | Statistical charts |
| Plotly | Latest | Interactive charts |
| Streamlit | Latest | Dashboard |

---

## Project Pipeline

### 1. Data Ingestion
- Loaded dataset using Apache Spark
- 1,470 rows and 35 columns
- Saved as Parquet

### 2. Data Preprocessing
- No missing values
- No duplicates
- Dropped 4 columns:
  EmployeeCount, StandardHours, Over18, EmployeeNumber
- Encoded categorical features
- Final dataset: 31 columns

### 3. Exploratory Data Analysis
- Attrition: 83.9% stayed, 16.1% left
- Sales has highest attrition (20.6%)
- Under 25 → 39.2% attrition
- Overtime → 30.5% attrition
- Low income → 28.6% attrition
- Strong predictors: TotalWorkingYears, JobLevel

### 4. Feature Engineering
Created:
- EngagementScore
- SatisfactionScore
- RiskScore
- ExperienceLevel
- IncomeLevel

### 5. Model Development

#### Attrition Prediction
- Random Forest Classifier
- Accuracy: 88.98%
- F1: 0.87
- Precision: 0.88
- Recall: 0.89

#### Performance Prediction
- Random Forest Regressor
- RMSE: 0.0208
- MAE: 0.0132
- R2: 0.9964

### 6. Clustering
- K-Means (K=3)
- Silhouette Score: 0.4052

Clusters:
- Mid-Career Stable
- Young High-Risk
- Senior High-Income

### 7. Insights & Recommendations

High-risk employees:
- 194 employees (13.2%)
- 87.1% from Young cluster

Recommendations:
1. Reduce overtime (max 10 hrs/week)
2. Increase low salaries
3. Focus on young employees
4. Improve engagement
5. Monitor high-risk employees monthly

---

## Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Attrition | F1 | 0.87 |
| Performance | R2 | 0.9964 |
| Clustering | Silhouette | 0.4052 |

---

## Cloud Platform

Databricks with Apache Spark

---

## Dashboard

https://hr-ai-system.streamlit.app/

---

## How to Run

### Install

git clone https://github.com/Mustafa-elsherif/HR-Analytics-AI-System.git  
cd HR-Analytics-AI-System  
pip install -r requirements.txt  

### Run

streamlit run src/dashboard.py

---

## License

Educational project for university course.