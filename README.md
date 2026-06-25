# Heart Disease Prediction — ML Project

Predicting the presence of heart disease from the **UCI Heart Disease** dataset using
classic machine learning models (KNN, Logistic Regression, Decision Tree, Random Forest).

## Project Structure

```
heart-disease-ml-project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/heart_disease_uci.csv
│   └── processed/            # cleaned & feature-engineered data
├── notebooks/                # step-by-step workflow (01 → 07)
├── figures/                  # generated charts
├── results/                  # model metrics
└── report/                   # LaTeX report
```

## Setup

```bash
pip install -r requirements.txt
```

---

## Steps

### ✅ Step 1 — Data Cleaning (`notebooks/01_data_cleaning.ipynb`)

**What it does (in order):**
1. Load dataset
2. Rows/columns (920 × 16)
3. Missing values table
4. Duplicates check (0 found)
5. Drop `id`
6. Binary `target` from `num` (>0 → 1)
7. Handle missing — numeric→median, categorical→mode
8. Clean categoricals — `fbs`/`exang`→int, text lowercased/stripped, `ca`→int
9. Before/after summary table
10. Save cleaned data

**Outputs:**
- `data/processed/heart_cleaned.csv` → 920 rows × 15 cols, 0 missing
- Target balance: 509 disease / 411 no-disease (nicely balanced)

**Figures:**
- `figures/missing_values.png` (bar chart)
- `figures/target_distribution.png` (count + %)
