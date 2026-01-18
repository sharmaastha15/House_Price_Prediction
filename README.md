# Housing Price Prediction Project

## 📌 Project Overview
This project implements a machine learning pipeline to predict housing prices (specifically the `MEDV` - Median value of owner-occupied homes) based on various features such as crime rate, number of rooms, and property age.

The workflow includes data cleaning, exploratory data analysis (EDA), feature scaling, and a comparative analysis of six different regression models.

## 🛠️ Technologies Used
* **Python** (Data Processing & Modeling)
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Visualization)
* **Scikit-Learn** (Machine Learning & Preprocessing)

## 📂 Dataset Description
The dataset contains **506 entries** and **14 columns**.
* **Target Variable:** `MEDV` (Median value of owner-occupied homes in $1000s).
* **Features:**
    * `CRIM`: Per capita crime rate by town.
    * `RM`: Average number of rooms per dwelling.
    * `AGE`: Proportion of owner-occupied units built prior to 1940.
    * `LSTAT`: % lower status of the population.
    * (And others including `ZN`, `INDUS`, `NOX`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`).

## ⚙️ Project Workflow

### 1. Data Cleaning
The raw dataset contained missing values in several columns (`CRIM`, `ZN`, `INDUS`, `CHAS`, `AGE`, `LSTAT`).
* **Handling Nulls:** Missing values were filled using the **mean** of their respective columns.
* **Duplicates:** Verified that no duplicate records existed.
* **Output:** The cleaned data is saved as `cleaned_dataset_superstore.csv`.

### 2. Exploratory Data Analysis (EDA)
* **Correlation Matrix:** Generated a heatmap to identify strong relationships between features (e.g., `RM` vs `MEDV` and `LSTAT` vs `MEDV`).
* **Boxplots:** Created to visualize outliers across different features.

### 3. Preprocessing
* **Splitting:** Data was split into 80% Training and 20% Testing sets.
* **Scaling:** Applied `StandardScaler` to normalize features, ensuring models like SVR and Linear Regression perform optimally.

### 4. Model Training & Evaluation
Six regression models were trained and tested:
1.  Linear Regression
2.  Ridge Regression
3.  Lasso Regression
4.  Random Forest Regressor
5.  Gradient Boosting Regressor
6.  Support Vector Regression (SVR)

## 📊 Results

The models were evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$).

| Model | MSE | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **7.36** | **2.71** | **1.89** | **0.8996** |
| Random Forest | 8.26 | 2.87 | 2.07 | 0.8874 |
| Linear Regression | 25.02 | 5.00 | 3.15 | 0.6589 |
| Ridge Regression | 25.02 | 5.00 | 3.15 | 0.6588 |
| Lasso Regression | 26.14 | 5.11 | 3.22 | 0.6436 |
| SVR | 26.47 | 5.15 | 2.78 | 0.6390 |

### Key Findings:
* **Best Performer:** **Gradient Boosting Regressor** achieved the highest accuracy with an $R^2$ of **~0.90**.
* **Runner Up:** Random Forest also performed exceptionally well ($R^2$ ~0.89).
* **Linear Models:** Simple linear models (Linear, Ridge, Lasso) struggled to capture the complexity of the data, hovering around 65% accuracy.

## 📈 Visualizations
The script generates the following plots:
1.  **Correlation Heatmap:** To see feature dependencies.
2.  **Feature Importance:** (Gradient Boosting) Identifies which factors drive prices the most.
3.  **Actual vs. Predicted Plot:** A scatter plot comparing true values against model predictions.
4.  **Residual Plot:** Analysis of error distribution to check for bias.

## 🚀 How to Run
1.  Ensure the dataset `HousingData.csv` is in the correct directory.
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Run the script:
    ```python
    python house_price_prediction.py
    ```
