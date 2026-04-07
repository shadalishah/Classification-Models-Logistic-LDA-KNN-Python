# 🔍 Classification Models — Stock Market Direction & Vehicle Efficiency Prediction

> **Skills Demonstrated:** Logistic Regression · LDA · QDA · Naive Bayes · KNN · Binary Classification · Confusion Matrix · Model Comparison · Python · Scikit-learn · Statsmodels

---

## 🎯 Project Overview

This project applies **five classification algorithms** to two real-world business problems:

1. **Weekly Stock Market Data** — Predicting whether the market goes **Up or Down** each week
2. **Auto Dataset** — Predicting whether a car has **high or low fuel efficiency** (binary classification)
3. **Boston Dataset** — Predicting whether a suburb has **above or below median crime rate**

Each model is trained, evaluated with a confusion matrix, and compared by test accuracy — following a rigorous train/test split methodology.

---

## 📁 Datasets Used

| Dataset | Source | Size | Target Variable |
|---------|--------|------|----------------|
| **Weekly** | S&P 500 Weekly Returns 1990–2010 (Real) | 1,089 rows, 9 features | Market Direction (Up/Down) |
| **Auto** | Carnegie Mellon StatLib (Real) | 397 rows, 9 features | mpg01 — High/Low Fuel Efficiency |
| **Boston** | U.S. Census Bureau (Real) | 506 rows, 13 features | crim_ab — Above/Below Median Crime |

---

## 🔧 Techniques & Tools Applied

| Technique | Library | Use Case |
|-----------|---------|----------|
| Logistic Regression (GLM) | `statsmodels` | Stock market direction & crime prediction |
| Linear Discriminant Analysis (LDA) | `sklearn` | When predictors are normally distributed |
| Quadratic Discriminant Analysis (QDA) | `sklearn` | When class covariances differ |
| Naive Bayes (Gaussian) | `sklearn` | Fast probabilistic baseline |
| K-Nearest Neighbors (KNN) | `sklearn` | Non-parametric, non-linear boundaries |
| Confusion Matrix | `ISLP` | Evaluating Type I & Type II errors |
| Train/Test Split | `sklearn` | Honest out-of-sample evaluation |

**Libraries:** `numpy` · `pandas` · `statsmodels` · `scikit-learn` · `matplotlib` · `seaborn` · `ISLP`

---

## 📊 Key Results

### Exercise 13 — Weekly Stock Market Direction Prediction

**Full model (all lag variables + Volume) — Training on 1990–2008, Testing on 2009–2010:**

Only **Lag2** was statistically significant among all predictors.

**Model Comparison on Held-Out Test Set (2009–2010, n=104):**

| Model | Confusion Matrix (Down/Up) | Test Accuracy | Recommendation |
|-------|---------------------------|---------------|----------------|
| Logistic Regression | TP=37, TN=20, FP=23, FN=24 | **54.8%** | ❌ Near random |
| **LDA** | TP=56, TN=9, FP=34, FN=5 | **🏆 62.5%** | ✅ Best model |
| QDA | TP=61, TN=0, FP=43, FN=0 | 58.7% | Predicts only "Up" |
| KNN (K=1) | TP=30, TN=22, FP=21, FN=31 | 50.0% | ❌ Random guessing |
| Naive Bayes | TP=61, TN=0, FP=43, FN=0 | 58.7% | Predicts only "Up" |

> **Winner: LDA with 62.5% accuracy** — outperforms all other methods because Lag2 approximately follows a normal distribution, satisfying LDA's assumptions. QDA and Naive Bayes both collapse to predicting only "Up", indicating the decision boundary is linear not quadratic.

---

### Exercise 14 — High vs Low Fuel Efficiency (mpg01 Binary Classification)

**Key Predictors Identified:** `weight`, `displacement`, `cylinders` (highest correlation with mpg01)

**Train/Test Split:** 75% training / 25% test (random_state=1)

| Model | Test Accuracy | Test Error | Confusion Matrix |
|-------|--------------|------------|-----------------|
| **QDA** | **93.9%** | **6.1%** | TN=49, TP=42, FP=2, FN=5 — tied best |
| **Logistic Regression** | **93.9%** | **6.1%** | Tied best |
| LDA | 92.9% | 7.1% | TN=49, TP=42, FP=5, FN=2 |
| Naive Bayes | 92.9% | 7.1% | Close to LDA |
| KNN (K=13) | **91.8%** | 8.2% | Best KNN configuration |
| KNN (K=1) | 84.7% | 15.3% | Overfits training data |

**KNN Accuracy by K value (top performers):**

| K | Accuracy |
|---|----------|
| K=1 | 84.7% |
| K=8 | 90.8% |
| **K=13** | **91.8%** ✅ |
| K=18 | 91.8% |
| K=20 | 91.8% |

> **Winner: QDA & Logistic Regression tied at 93.9%** — weight, displacement, and cylinders together are powerful predictors of fuel efficiency class. KNN peaks at K=13 showing the optimal bias-variance tradeoff for this dataset.

---

### Exercise 16 — Boston Crime Rate Classification

**Key Predictors Selected:** `indus`, `nox`, `age`, `dis`, `rad`, `tax`

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| **KNN (K=1)** | **93.7%** 🏆 | Best overall |
| KNN (K=2) | 92.1% | Strong performance |
| Logistic Regression | 86.6% | Good parametric baseline |
| LDA | 81.9% | Lower — non-linear boundary |
| Naive Bayes | 81.1% | Similar to LDA |

**KNN top results (Boston crime):**

| K | Accuracy |
|---|----------|
| **K=1** | **93.7%** 🏆 |
| K=2,3 | 92.1% |
| K=4,6 | 91.3% |

> **Winner: KNN (K=1) at 93.7%** — unlike the Weekly dataset, crime prediction benefits from non-linear decision boundaries. High-crime suburbs cluster geographically and industrially, making nearest-neighbor methods highly effective.

---

## 💡 Business Insights

1. **Stock Market Prediction is Hard:** Even the best model (LDA, 62.5%) barely beats random chance (50%) on weekly returns — confirming the Efficient Market Hypothesis. Traders should be skeptical of ML-based short-term prediction claims.

2. **Vehicle Efficiency Classification:** QDA and Logistic Regression both achieve ~94% accuracy in classifying high vs low efficiency cars — practical for automotive industry fleet analysis and environmental compliance screening.

3. **Crime Rate Profiling:** KNN at K=1 achieves 93.7% accuracy in identifying high-crime Boston suburbs using industrial, pollution, and accessibility features — a useful baseline for urban planning risk assessment models.

4. **Algorithm Selection Matters:** LDA outperforms QDA when the decision boundary is linear (stock data). KNN outperforms LDA when the boundary is non-linear (crime data). Choosing the right model for the data structure is more impactful than parameter tuning.

---

## 🗂️ File Structure

```
Chapter_4_Applied_Exercise_Solutions/
│
├── Chapter_4.ipynb          ← Main analysis notebook (all exercises)
├── chapter_4.html           ← Rendered HTML version (easy browser viewing)
├── chapter_4.qmd            ← Quarto source file
└── README.md                ← This file
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install ISLP scikit-learn statsmodels pandas numpy matplotlib seaborn

# Launch notebook
jupyter notebook Chapter_4.ipynb
```

---

## 📚 Reference

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
*An Introduction to Statistical Learning with Applications in Python.* Springer.
Chapter 4: Classification — Applied Exercises 13–16.

---

## 🙏 Acknowledgements

Special thanks to **Karim Aboussel Ham** whose repository [ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM) provided useful guidance and reference during the completion of this project.

---

## 👤 About the Author

**Shad Ali Shah**
🎓 MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad
💡 Passionate about the intersection of **Economics**, **Data Science**, and **Machine Learning**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/shadalishah)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shadalishah)

---

*Part of the [ML Portfolio](../README.md) by Shad Ali Shah*
