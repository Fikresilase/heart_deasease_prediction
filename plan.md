Here is the complete updated plan based on the real cleaned dataset.

# Project Title

**Early Prediction of Heart Disease Using Machine Learning Classification Models**

Goal:

```text
Predict whether a patient has heart disease or not.
```

Target:

```text
target = 0 -> No Heart Disease
target = 1 -> Heart Disease
```

Current cleaned dataset:

```text
File: data/processed/heart_cleaned.csv
Rows: 920
Columns: 15
Missing values: 0

Target distribution:
0 -> 411 patients
1 -> 509 patients
```

Real dataset columns:

```text
Numerical columns:
age, trestbps, chol, thalch, oldpeak

Binary columns:
fbs, exang, target

Categorical columns:
sex, dataset, cp, restecg, slope, thal

Ordinal/discrete column:
ca
```

Column meanings:

```text
age      -> Patient age
sex      -> Patient sex
dataset  -> Source dataset location
cp       -> Chest pain type
trestbps -> Resting blood pressure
chol     -> Serum cholesterol
fbs      -> Fasting blood sugar
restecg  -> Resting ECG result
thalch   -> Maximum heart rate achieved
exang    -> Exercise-induced angina
oldpeak  -> ST depression induced by exercise
slope    -> Slope of peak exercise ST segment
ca       -> Number of major vessels
thal     -> Thalassemia result
target   -> Binary heart disease diagnosis
```

---

# 1. Repository Structure

```text
heart-disease-ml-project/
|
|-- README.md
|-- requirements.txt
|
|-- data/
|   |-- raw/
|   |   |-- heart_disease_uci.csv
|   |-- processed/
|       |-- heart_cleaned.csv
|       |-- X_train.csv
|       |-- X_test.csv
|       |-- y_train.csv
|       |-- y_test.csv
|       |-- heart_train.csv
|       |-- heart_test.csv
|       |-- heart_final.csv
|
|-- notebooks/
|   |-- 01_data_cleaning.ipynb
|   |-- 02_data_splitting.ipynb
|   |-- 03_eda_and_feature_engineering.ipynb
|   |-- 04_knn.ipynb
|   |-- 05_logistic_regression.ipynb
|   |-- 06_decision_tree.ipynb
|   |-- 07_random_forest.ipynb
|   |-- 08_model_comparison.ipynb
|
|-- figures/
|   |-- target_distribution.png
|   |-- age_distribution.png
|   |-- cholesterol_distribution.png
|   |-- blood_pressure_distribution.png
|   |-- max_heart_rate_distribution.png
|   |-- chest_pain_target.png
|   |-- sex_target.png
|   |-- dataset_target.png
|   |-- correlation_heatmap.png
|   |-- feature_importance.png
|   |-- roc_curves.png
|   |-- model_comparison.png
|   |-- confusion_matrix_knn.png
|   |-- confusion_matrix_logistic.png
|   |-- confusion_matrix_tree.png
|   |-- confusion_matrix_forest.png
|
|-- results/
|   |-- knn_metrics.csv
|   |-- logistic_metrics.csv
|   |-- decision_tree_metrics.csv
|   |-- random_forest_metrics.csv
|   |-- final_model_comparison.csv
|
|-- models/
|   |-- preprocessing_pipeline.pkl
|   |-- random_forest_model.pkl
|
|-- report/
    |-- report.tex
    |-- references.bib
```

---

# 2. Serial Workflow

## Step 1: `01_data_cleaning.ipynb`

Do:

```text
Load raw UCI heart disease dataset
Check rows and columns
Check missing values
Check duplicates
Drop irrelevant columns such as id
Create binary target from original diagnosis column
Handle missing values
Clean categorical values
Save cleaned dataset
```

Output:

```text
data/processed/heart_cleaned.csv
```

Expected cleaned dataset:

```text
920 rows
15 columns
No missing values
Binary target column already created
```

Visuals:

```text
Missing value bar chart
Target distribution chart
Before/after cleaning summary table
```

---

## Step 2: `02_data_splitting.ipynb`

Do:

```text
Load data/processed/heart_cleaned.csv
Separate features X from target y
Use stratified train/test split
Keep the test set untouched until final model evaluation
Use only the training set for cross-validation and model selection
Save train and test files
```

Recommended split:

```text
Train size: 80%
Test size: 20%
Stratify by target
Random state: 42
```

Outputs:

```text
data/processed/X_train.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_test.csv
data/processed/heart_train.csv
data/processed/heart_test.csv
```

Why this step matters:

```text
It prevents data leakage.
It makes cross-validation fair.
It keeps the final test results honest.
```

---

## Step 3: `03_eda_and_feature_engineering.ipynb`

Do:

```text
Load heart_train.csv for EDA
Explore target distribution in the training data
Explore numerical columns: age, trestbps, chol, thalch, oldpeak
Explore categorical columns: sex, dataset, cp, restecg, slope, thal
Compare all important features against target
Build preprocessing pipeline
One-hot encode categorical columns
Scale numerical columns
Pass binary/discrete columns cleanly
Fit preprocessing only on training data
Transform both training and test data
Save final processed dataset
Save preprocessing pipeline
```

Numerical features:

```text
age
trestbps
chol
thalch
oldpeak
```

Categorical features:

```text
sex
dataset
cp
restecg
slope
thal
```

Binary/discrete features:

```text
fbs
exang
ca
```

Visuals:

```text
Correlation heatmap
Age vs heart disease
Chest pain type vs heart disease
Sex vs heart disease
Dataset source vs heart disease
Cholesterol distribution
Resting blood pressure distribution
Maximum heart rate distribution
Oldpeak distribution
Feature importance preview
```

Outputs:

```text
data/processed/heart_final.csv
models/preprocessing_pipeline.pkl
```

Important rule:

```text
Do not fit encoders or scalers on the full dataset.
Fit them only on X_train, then transform X_train and X_test.
```

---

## Step 4: `04_knn.ipynb`

Do:

```text
Load training and test data
Train KNN
Try k values from 1 to 20
Use cross-validation on the training set
Choose the best k
Evaluate the final KNN model on the untouched test set
Save metrics
```

Visuals:

```text
Accuracy vs K graph
Confusion matrix
ROC curve
```

Output:

```text
results/knn_metrics.csv
figures/confusion_matrix_knn.png
```

---

## Step 5: `05_logistic_regression.ipynb`

Do:

```text
Load training and test data
Train Logistic Regression
Use cross-validation on the training set
Evaluate on the untouched test set
Interpret coefficients
Save metrics
```

Visuals:

```text
Confusion matrix
ROC curve
Coefficient importance chart
```

Output:

```text
results/logistic_metrics.csv
figures/confusion_matrix_logistic.png
```

---

## Step 6: `06_decision_tree.ipynb`

Do:

```text
Load training and test data
Train Decision Tree
Tune max_depth using cross-validation on the training set
Evaluate best tree on the untouched test set
Save metrics
```

Visuals:

```text
Decision tree diagram
Confusion matrix
ROC curve
Feature importance chart
```

Output:

```text
results/decision_tree_metrics.csv
figures/confusion_matrix_tree.png
```

---

## Step 7: `07_random_forest.ipynb`

Do:

```text
Load training and test data
Train Random Forest
Use GridSearchCV on the training set
Evaluate best Random Forest model on the untouched test set
Save final model
Save metrics
```

Visuals:

```text
Feature importance chart
Confusion matrix
ROC curve
```

Output:

```text
results/random_forest_metrics.csv
figures/confusion_matrix_forest.png
models/random_forest_model.pkl
```

This is the main model.

---

## Step 8: `08_model_comparison.ipynb`

Do:

```text
Load all saved metrics
Compare all models using the same test set
Compare cross-validation scores
Pick the best model
Create final result visuals
Explain why the best model performed best
```

Visuals:

```text
Model comparison bar chart
All ROC curves in one graph
Final results table
```

Output:

```text
results/final_model_comparison.csv
figures/model_comparison.png
figures/roc_curves.png
```

---

# 3. Metrics to Use

For every model, calculate:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
Cross-validation mean score
Cross-validation standard deviation
```

Final comparison table:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Mean | CV Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.832 | 0.851 | 0.843 | 0.847 | 0.913 | 0.827 | 0.035 |
| Random Forest | 0.810 | 0.832 | 0.824 | 0.828 | 0.905 | 0.833 | 0.036 |
| KNN | 0.848 | 0.843 | 0.892 | 0.867 | 0.902 | 0.851 | 0.028 |
| Decision Tree | 0.761 | 0.774 | 0.804 | 0.788 | 0.756 | 0.760 | 0.021 |

Result interpretation:

```text
Logistic Regression was best by ROC-AUC.
KNN was best by accuracy and F1-score.
Random Forest was strong but not the best overall model.
Decision Tree performed weakest, which is expected because a single tree is less stable than ensemble or distance/linear methods.
```

Main evaluation rule:

```text
Use cross-validation for model selection.
Use the test set only once for final model evaluation.
```

---

# 4. Report Structure

Your LaTeX report should look like this:

```text
1. Abstract
2. Introduction
3. Problem Statement
4. Dataset Description
5. Data Cleaning
6. Train/Test Split Strategy
7. Exploratory Data Analysis
8. Feature Engineering
9. Machine Learning Models
   9.1 KNN
   9.2 Logistic Regression
   9.3 Decision Tree
   9.4 Random Forest
10. Experimental Setup
11. Results and Discussion
12. Model Comparison
13. Conclusion
14. Future Work
15. References
```

---

# 5. Best Diagrams for the Report

Include these:

```text
1. Project workflow diagram
2. Dataset cleaning pipeline
3. Train/test split diagram
4. Target distribution chart
5. Correlation heatmap
6. Feature importance chart
7. Confusion matrix for each model
8. ROC curve comparison
9. Model performance bar chart
```

The most impressive ones:

```text
ROC curve comparison
Random Forest feature importance
Correlation heatmap
Model comparison bar chart
Train/test/cross-validation workflow diagram
```

---

# 6. Presentation Story

Use this flow:

```text
Problem:
Heart disease is dangerous and early prediction is valuable.

Data:
The cleaned dataset contains 920 patient records and 15 columns from multiple UCI heart disease sources.

Cleaning:
Missing values, duplicates, irrelevant columns, and original diagnosis labels were handled.

Splitting:
The data was split into training and test sets using stratification so both classes stayed balanced.

Feature Engineering:
Categorical medical features were encoded, numerical features were scaled, and preprocessing was fitted only on training data.

Models:
KNN, Logistic Regression, Decision Tree, Random Forest.

Results:
Logistic Regression achieved the best ROC-AUC, KNN achieved the best accuracy and F1-score, Random Forest also performed strongly, and Decision Tree was the weakest model.

Insight:
Chest pain type, maximum heart rate, exercise-induced angina, oldpeak, thalassemia result, and major vessels were important predictors.

Conclusion:
Machine learning can support early heart disease risk prediction.
```

---

# 7. Final Slide / Final Report Message

Use this as your core conclusion:

```text
Among the four models tested, Logistic Regression achieved the best ROC-AUC, while KNN achieved the best accuracy and F1-score. Random Forest also performed strongly, but it was not the overall best model in this experiment. These results suggest that the cleaned heart disease dataset is small, structured, and well-preprocessed enough for simpler models to perform very well. The experiment used a stratified train/test split, cross-validation for model selection, and final evaluation on an untouched test set. The project demonstrates that machine learning can be used as a decision-support tool for early heart disease prediction.
```

This plan is stronger than the first version because it is based on the real cleaned dataset, uses a proper train/cross-validation/test workflow, and reports the actual model results honestly.



