# 📊 Engage2Value: From Clicks to Conversions

### Predicting Customer Purchase Value from Multi-Session Digital Behavior

---

**Course:** Machine Learning Practice Project (May 2025 Term)
**Problem Type:** Supervised Regression
**Platform:** Kaggle
**Author:** **Tanush Sudheer Tambe**

---

## 🧠 Problem Statement

The goal of this project is to **predict a customer’s purchase value** based on their **multi-session behavior across digital touchpoints**.

The dataset captures detailed session-level interactions including:

* Page views
* Total hits
* Traffic source
* Device & browser information
* Geo-location
* Temporal behavior

The target variable **`purchaseValue`** is:

* Extremely **right-skewed**
* Highly **zero-inflated**
* Influenced by **complex non-linear interactions**

---

## 🎯 Key Challenges

* Heavy **class imbalance** (majority zero purchase sessions)
* Extremely **right-skewed target** (skew > 50)
* High-cardinality categorical features
* Risk of **data leakage via user/session IDs**
* Multicollinearity among strong behavioral features

---

## 🧪 Dataset Overview

| Dataset | Rows    | Columns |
| ------- | ------- | ------- |
| Train   | 116,023 | 52      |
| Test    | 29,006  | 51      |

* Target: `purchaseValue`
* Mixed data types: numerical, categorical, boolean, timestamps
* Large number of **missing and meaningless values**

---

## 🔍 Exploratory Data Analysis (EDA)

### Numerical Insights

* **Strongest correlated features:**

  * `totalHits`
  * `pageViews`
  * `sessionNumber`
* Target has:

  * Median = 0
  * 75th percentile = 0
  * Extremely large outliers

### Target Transformation

| Method         | Skew                   |
| -------------- | ---------------------- |
| Original       | 53.9                   |
| Power (1/1.54) | 12.15                  |
| log1p          | 1.46 (over-compressed) |

➡️ **Power transformation with exponent = 1/1.54** was chosen as the best trade-off.

---

## 🧹 Data Cleaning Strategy

### Removed:

* Columns with **100% null values**
* Constant features (single unique value)
* Features containing only `"not available in demo dataset"`

### Imputation Rules:

* Boolean → logical defaults
* Location → `(not set)`
* Categorical → `(not provided)` / `Other`
* Numerical → mean imputation (pipeline)

---

## ⚙️ Feature Engineering

### 1️⃣ Session & Interaction Features

* `hits_per_pageview`
* `pageviews_per_hour`
* `session_page_product`
* `bounce_hit_ratio`
* `session_per_hit`

### 2️⃣ Behavioral Indicators

* `is_repeat_visitor`
* `is_video_ad_and_bounce`

### 3️⃣ Frequency Encoding

* Browser
* City
* Campaign

### 4️⃣ Time-Based Features

* Business hour indicators
* Weekend / weekday flags
* Peak month / quarter indicators

---

## 🧪 Leakage-Aware Modeling

Two parallel datasets were maintained:

| Dataset      | Description                          |
| ------------ | ------------------------------------ |
| Leakage-free | Excludes `userId`, `sessionId`       |
| ID-inclusive | Includes ID-based frequency features |

➡️ This allowed **controlled comparison** between realistic and leaderboard-optimized models.

---

## 🔧 Preprocessing Pipeline

Implemented using **Scikit-Learn Pipelines**:

### Numerical

* Mean Imputation
* RobustScaler

### Categorical

* **Low Cardinality:** OneHotEncoder
* **High Cardinality:** TargetEncoder

All outputs were converted to **fully numeric, float64 matrices**.

---

## 🧠 Post-Pipeline Feature Engineering

Derived from scaled features:

* `pageviews_business_hour_scaled`
* `hits_weekend_scaled`
* `efficiency_business`
* `repeat_hits`

---

## ✂️ Feature Selection

Correlation-based trimming:

* Retained features with |corr| > 0.01 (no IDs)
* |corr| > 0.07 (ID-inclusive)

This reduced noise and stabilized training.

---

## 🧩 Feature Stacking

### 1️⃣ K-Means Clustering

* Applied on trimmed, scaled data
* Optimal clusters = **3**
* New feature: `cluster`

### 2️⃣ Binary Purchase Classifier

* Model: **LightGBM Classifier**
* Task: Predict `purchaseValue > 0`
* Output: `buy_prob`

**Validation Performance**

* Accuracy: **0.963**
* Precision: **0.881**
* Recall: **0.950**
* F1 Score: **0.914**

➡️ `buy_prob` became the **most correlated feature** with the target.

---

## 🤖 Models Trained

### Baseline Models

* LightGBM Regressor
* XGBoost Regressor
* ExtraTrees Regressor

### Observations

| Model      | Behavior           |
| ---------- | ------------------ |
| LightGBM   | Underfitting       |
| XGBoost    | Overfitting        |
| ExtraTrees | Severe overfitting |

---

## 🔍 Hyperparameter Tuning

### Tuned Using:

* `RandomizedSearchCV`
* 3-fold CV
* R² scoring

### Best Single Models

* **Tuned LightGBM (No IDs):** ~0.70 public R²
* **Tuned LightGBM (With IDs):** ~0.728 public R² (leakage)
* **Tuned XGBoost:** Poor generalization

---

## 🧠 Ensemble Learning

### VotingRegressor (5 LightGBMs)

* Slight parameter variations
* No ID leakage
* Best balance of bias & variance

➡️ Achieved **~0.70 public leaderboard R²** without relying on IDs.

---

## 📤 Submission Strategy

* Final model trained on full train + validation set
* Power-inverse transformation applied
* Small predictions clipped to zero to reduce noise
* Output saved as `submission.csv`

---

## 📈 Final Results (Kaggle)

| Model                     | Public R²  |
| ------------------------- | ---------- |
| Tuned LightGBM (No IDs)   | ~0.698     |
| Tuned LightGBM (With IDs) | **~0.728** |
| Ensemble (No IDs)         | **~0.70**  |

---

## 🏁 Key Takeaways

* Target transformation is **critical** for skewed regression
* Feature stacking (cluster + buy_prob) adds **major signal**
* ID features improve leaderboard score but introduce leakage
* Ensemble models provide best **real-world generalization**

---

## 🧑‍💻 How to Run

This project is implemented as a **single Kaggle notebook**.

1. Open notebook on Kaggle
2. Attach competition dataset
3. Run cells sequentially
4. Submission file is generated automatically

---

## 📜 License

This project is released under the **MIT License** and is intended for **educational and research purposes**.

---

Just tell me 👍
