# Heart Disease Prediction - ML Project

Predicting the presence of heart disease from the UCI Heart Disease dataset using classic machine learning models: KNN, Logistic Regression, Decision Tree, and Random Forest.

## Project Structure

```text
heart-disease-ml-project/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/heart_disease_uci.csv
|   |-- processed/
|       |-- heart_cleaned.csv
|       |-- X_train.csv
|       |-- X_test.csv
|       |-- y_train.csv
|       |-- y_test.csv
|       |-- heart_train.csv
|       |-- heart_test.csv
|       |-- heart_final.csv
|-- notebooks/
|   |-- 01_data_cleaning.ipynb
|   |-- 02_data_splitting.ipynb
|   |-- 03_eda_and_feature_engineering.ipynb
|   |-- 04_knn.ipynb
|   |-- 05_logistic_regression.ipynb
|   |-- 06_decision_tree.ipynb
|   |-- 07_random_forest.ipynb
|   |-- 08_model_comparison.ipynb
|-- figures/
|-- results/
|-- models/
|-- report/
```

## Setup and Run

Open PowerShell inside the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name heart-disease-venv --display-name "Python (heart disease)"
```

After that, open a notebook in VS Code or Jupyter and select the kernel:

```text
Python (heart disease)
```

Do not type `Python (heart disease)` in the terminal. It is the notebook kernel name, so choose it from the notebook kernel menu.

## Notebook Order

Run the notebooks in this order:

```text
01_data_cleaning.ipynb
02_data_splitting.ipynb
03_eda_and_feature_engineering.ipynb
04_knn.ipynb
05_logistic_regression.ipynb
06_decision_tree.ipynb
07_random_forest.ipynb
08_model_comparison.ipynb
```

## Completed Steps

### Step 1 - Data Cleaning (`notebooks/01_data_cleaning.ipynb`)

What it does:

1. Load dataset
2. Check rows and columns
3. Check missing values
4. Check duplicates
5. Drop `id`
6. Create binary `target` from `num`
7. Handle missing values
8. Clean categorical values
9. Save cleaned data

Outputs:

```text
data/processed/heart_cleaned.csv
```

Cleaned dataset summary:

```text
Rows: 920
Columns: 15
Missing values: 0
Target 0: 411
Target 1: 509
```

### Step 2 - Data Splitting (`notebooks/02_data_splitting.ipynb`)

What it does:

1. Load `data/processed/heart_cleaned.csv`
2. Separate features `X` from target `y`
3. Create stratified train/test split
4. Save train and test files

Outputs:

```text
data/processed/X_train.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_test.csv
data/processed/heart_train.csv
data/processed/heart_test.csv
```

Split summary:

```text
Train: 736 rows
Test: 184 rows
```

### Step 3 - EDA and Feature Engineering (`notebooks/03_eda_and_feature_engineering.ipynb`)

What it does:

1. Load `heart_train.csv` for EDA
2. Explore numerical and categorical features
3. Compare important features against target
4. One-hot encode categorical columns
5. Scale numerical columns
6. Pass binary/discrete columns cleanly
7. Fit preprocessing only on training data
8. Transform both training and test data
9. Save final processed data and preprocessing pipeline

Outputs:

```text
data/processed/heart_final.csv
models/preprocessing_pipeline.pkl
```
