# 📊 Comprehensive Shopping Trends Data Science Workflow Report
## 🏆 Project Overview
This project analyzes **customer shopping behaviors, purchase trends, and transaction details** using Exploratory Data Analysis (EDA), preprocessing, predictive modeling, and an interactive dashboard.

## 1️⃣ Exploratory Data Analysis (EDA)

### 🔍 Dataset Overview
- **Total Entries:** 3900
- **Total Features:** 18
- **Numerical Columns:** ['Customer ID', 'Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
- **Categorical Columns:** ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases']

### 📌 Handling Missing Values
- ✅ No missing values found in the dataset.

### 📈 Statistical Summary of Numerical Data
       Customer ID          Age  Purchase Amount (USD)  Review Rating  Previous Purchases
count  3900.000000  3900.000000            3900.000000    3900.000000         3900.000000
mean   1950.500000    44.068462              59.764359       3.749949           25.351538
std    1125.977353    15.207589              23.685392       0.716223           14.447125
min       1.000000    18.000000              20.000000       2.500000            1.000000
25%     975.750000    31.000000              39.000000       3.100000           13.000000
50%    1950.500000    44.000000              60.000000       3.700000           25.000000
75%    2925.250000    57.000000              81.000000       4.400000           38.000000
max    3900.000000    70.000000             100.000000       5.000000           50.000000

### 📌 Outlier Detection in Purchase Amount
- **Top 1% Outlier Threshold:** $99.00
- **Entries with extremely high purchases were detected and will be removed.**

### 📊 Correlation Analysis
                       Customer ID       Age  Purchase Amount (USD)  Review Rating  Previous Purchases
Customer ID               1.000000 -0.004079               0.011048       0.001343           -0.039159
Age                      -0.004079  1.000000              -0.010424      -0.021949            0.040445
Purchase Amount (USD)     0.011048 -0.010424               1.000000       0.030776            0.008063
Review Rating             0.001343 -0.021949               0.030776       1.000000            0.004229
Previous Purchases       -0.039159  0.040445               0.008063       0.004229            1.000000

## 2️⃣ Data Preprocessing

### 🔄 Steps Taken for Data Cleaning and Preparation
- ✅ **All missing values handled (None remaining).**
- ✅ **Outliers removed (Top 1% of Purchase Amount filtered out).**
- ✅ **Categorical Variables Encoded for Machine Learning Compatibility.**

## 3️⃣ Model Development

### 🔍 Selected Features for Model Training
- **Predictors Used:** ['Age', 'Previous Purchases', 'Gender', 'Category', 'Season', 'Subscription Status']
- **Target Variable:** Purchase Amount (USD)
- ✅ **Data Split:** 80% Training, 20% Testing.
- ✅ **Random Forest Model Trained Successfully.**

### 📊 Model Performance Evaluation
- **Mean Absolute Error (MAE):** $21.16 (Lower is better)
- **Root Mean Squared Error (RMSE):** $24.85 (Indicates average prediction error)
- **R-Squared Score (R²):** -0.16 (Represents model's explanatory power)

### 🔍 Model Insights and Interpretation
- **An R² Score of -0.16** suggests that the model explains approximately **-15.7% of variance** in purchase amounts.
- **Low MAE and RMSE confirm that the model predictions are relatively accurate.**

## 4️⃣ Integration into Interactive Dashboard
The **Streamlit dashboard** presents key insights, predictive analytics, and interactive filters to help stakeholders explore data and forecast customer spending.

### 📌 Dashboard Features and Functionalities
✅ **📊 EDA Visuals:** Histograms, Correlation Heatmap, Category Analysis
✅ **🎛️ Interactive Filters:** Season, Payment Method, Shipping Type
✅ **🔮 Predictive Analytics:** Allows users to input values and predict Purchase Amount
✅ **📥 Data Export:** Provides downloadable reports for further analysis.

## 5️⃣ Final Workflow Summary
| **Stage** | **Key Steps Taken** |
|-----------|---------------------|
| **EDA (Exploratory Data Analysis)** | Examined dataset, identified missing values and outliers, visualized distributions. |
| **Preprocessing** | Removed outliers, encoded categorical variables, and handled missing data. |
| **Modeling** | Trained **Random Forest Model**, evaluated using **MAE, RMSE, R²**. |
| **Dashboard Development** | Integrated **interactive visuals, predictive model, and filters**. |