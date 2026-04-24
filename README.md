# HR Analytics AI System

## An AI-Powered System for Employee Analytics and Decision Support

---

## 📌 Project Overview
This project develops an end-to-end AI-powered HR analytics system using **Apache Spark** and **Machine Learning** to analyze employee data, predict attrition, evaluate performance, segment employees, and recommend targeted HR strategies.

The system processes the IBM HR Analytics Dataset containing 1,470 employees and 35 features, delivering actionable insights to support data-driven HR decision-making.

---

## 👥 Team Members

| Name | Student ID | Responsibilities |
|:---|:---|:---|
| **Mustafa Nabil** | 240103415 | Model Development, Clustering, Dashboard |
| **Mahmoud Hesham** | 240101375 | Model Development, Clustering, Dashboard |
| **Mariam Magdy** | 240102374 | Exploratory Data Analysis (EDA) |
| **Omar Ahmed Wafik** | 240101244 | Data Preprocessing, Feature Engineering |
| **Mohamed Nour** | 240100780 | Data Ingestion, Insights and HR Recommendations |

---

## 📁 Project Structure
```text
HR-Analytics-AI-System/
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv    # Raw IBM HR Dataset
│   ├── hr_data.parquet                          # Ingested data
│   ├── hr_data_preprocessed.parquet            # Preprocessed data
│   └── hr_data_engineered.parquet              # Feature engineered data
├── notebooks/
│   ├── 01_data_ingestion.ipynb                 # Data loading with Spark
│   ├── 02_data_preprocessing.ipynb             # Cleaning and encoding
│   ├── 03_eda.ipynb                            # Exploratory data analysis
│   ├── 04_feature_engineering.ipynb            # Feature creation
│   ├── 05_model_development.ipynb              # ML model training
│   ├── 06_clustering.ipynb                     # K-Means segmentation
│   └── 07_insights.ipynb                       # Insights and recommendations
├── src/
│   └── dashboard.py                            # Streamlit dashboard
├── models/
│   ├── attrition_model/                        # Saved classification model
│   └── performance_model/                      # Saved regression model
├── outputs/                                    # Visualizations & Charts
├── requirements.txt
└── README.md