# RegrLib – Guided Regression Framework for Learning and Research

**RegrLib** is a lightweight, educational-focused regression framework that automates the end-to-end regression process. It provides smart preprocessing, model selection, regularization tuning, and visualization — ideal for students, researchers, and anyone exploring regression techniques.

---

## 🚀 Features

- ✅ Linear, Polynomial, Ridge, and Lasso regression models
- 📐 Automatic polynomial degree selection
- 🎛️ Regularization tuning (manual + grid search)
- 🧼 Categorical encoding, feature scaling, missing value handling
- 📊 Regression metrics: MSE, R², R²adj
- 📈 Visualizations: prediction plots and residual analysis
- 🧠 Framework-style design: defines how you work, not just what you use

---

## ⚙️ Framework Logic and Workflow

RegrLib is a **framework** – not just a library. It guides your workflow, makes informed decisions, and automates best practices. Here's the logic:

### 🔄 Step 1: Data Preprocessing

- Automatically detects **categorical features** → applies one-hot encoding
- Handles **missing values** with simple imputation (mean/median)
- Applies **feature scaling** if needed (for regularized models)
- Optionally generates **polynomial features** (manual or auto degree selection)

---

### 🧠 Step 2: Intelligent Model Guidance

- If features are non-linear, avoids plain linear regression unless explicitly requested
- If multicollinearity is detected, prefers **Ridge Regression**
- If sparse solutions are needed, recommends **Lasso Regression**
- If `auto_model_select=True`, framework will choose the best model based on data patterns

---

### 📐 Step 3: Polynomial Degree Selection

- Framework can automatically select the best polynomial degree
- For each degree (e.g., 1 to 5):
  - Transforms data into polynomial features
  - Trains a model and evaluates it on validation data
- Chooses the degree with best performance (lowest MSE or highest R²)

---

### 🧪 Step 4: Regularization Tuning

- Supports both **manual** and **grid-based** alpha tuning
- Evaluates each alpha on validation set and selects the best one
- Works with Ridge and Lasso models

---

### 📊 Step 5: Evaluation and Visualization

- Computes metrics like **MSE, R², R²adj**
- Automatically generates:
  - **Prediction vs. actual plots**
  - **Residual plots** for error analysis

---

## 📁 Project Structure

```txt
regrlib/
├── models/           # Regression models (Linear, Ridge, Lasso, Polynomial)
├── preprocessing/    # One-hot encoding, scaling, polynomial features
├── tuning/           # Manual and grid search logic
├── metrics/          # MSE, R², R²adj
├── visualization/    # Plotting utilities
├── pipeline/         # High-level RegressionPipeline class
└── README.md
```
## 📦 Installation

Clone this repo:

```bash
git clone https://github.com/yourusername/regrlib.git
cd regrlib
```

