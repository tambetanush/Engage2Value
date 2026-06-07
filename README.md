# 📊 Engage2Value: From Clicks to Conversions

![Dashboard Preview](static/screenshots/1%20dashboard%201.png)

**Predicting Customer Purchase Value from Multi-Session Digital Behavior**

An end-to-end Machine Learning project that began as a competitive modeling task and evolved into a fully interactive, enterprise-grade ML Interpretability Dashboard.

**Course:** Machine Learning Practice Project (May 2025 Term)  
**Platform:** Kaggle & Custom Web Application  
**Author:** Tanush Sudheer Tambe

---

## 🌟 Application Showcase

To make the machine learning model accessible to business stakeholders, we built a highly interactive web application using **Flask, Plotly, and TailwindCSS (Crimson Theme)**. The dashboard provides real-time ML interpretability and inference.

### 1. Interactive Dashboard & Real-Time Insights
The landing page provides immediate insights into global behavioral drivers. It automatically generates **Partial Dependence Plots (PDP)** to demonstrate non-linear plateau effects and runs **Isolation Forests** to highlight anomalous high-frequency sessions in red.
![Dashboard Insights](static/screenshots/2%20dashboard%202.png)

### 2. Global Feature Importance (SHAP)
Navigate to the Data Exploration tab to see the top 10 features driving the predictive model globally across all segments.
![Global SHAP](static/screenshots/3%20data%20analytics%20global%20shap.png)

### 3. Deep Feature Analysis
Select any feature to instantly view its statistical inferences, probability distributions, outlier boxplots, and categorical segmentations. The app uses background algorithms to automatically discover and highlight highly profitable sub-cohorts.
![Data Analysis](static/screenshots/4%20data%20analysis.png)
![Advanced Data Analysis](static/screenshots/5%20data%20analysis%20advanced.png)

### 4. Bivariate Analysis & Marginal Effects
Uncover hidden interactions by selecting secondary features. The app generates **2D Interaction Heatmaps**, grouped scatter plots, and isolated Marginal Effects.
![PDP and Bivariate](static/screenshots/6%20pdp%20and%20bivariate.png)

### 5. Live Purchase Prediction Engine
A dynamic form that allows you to simulate customer behavior. Input session parameters, hit "Randomize" to test logic, and instantly generate the expected purchase value using our robust stacked ensemble pipeline.
![Prediction Form](static/screenshots/7%20predict%20form.png)
![Prediction Result](static/screenshots/8%20prediction.png)

---

## 🧠 Machine Learning Architecture (Kaggle Task)

### Problem Statement
The goal of this project is to **predict a customer’s purchase value** based on their multi-session behavior across digital touchpoints. The target variable `purchaseValue` is:
- Extremely **right-skewed** (skew > 50)
- Highly **zero-inflated** (majority zero purchase sessions)
- Influenced by **complex non-linear interactions**

### Dataset Overview
| Dataset | Rows    | Columns |
| ------- | ------- | ------- |
| Train   | 116,023 | 52      |
| Test    | 29,006  | 51      |

### Feature Engineering & Data Pipeline
- **Target Transformation:** Power transformation with exponent = 1/1.54 was chosen as the best trade-off to handle extreme skewness.
- **Categorical Processing:** Handled high-cardinality features using `TargetEncoder` and low-cardinality with `OneHotEncoder`.
- **Behavioral Indicators:** Engineered powerful features like `bounce_hit_ratio`, `session_per_hit`, and `is_repeat_visitor`.
- **Feature Stacking:** 
  1. Applied **K-Means Clustering** to grouped numerical data.
  2. Trained a secondary **LightGBM Binary Classifier** to predict `purchaseValue > 0`. The resulting `buy_prob` became the most correlated feature with the target.

### Model Ensembling
We utilized a **VotingRegressor** stacking 5 optimized LightGBM models with slight parameter variations. This achieved the best balance of bias and variance, resulting in a **~0.70 public leaderboard R²** without relying on ID leakage.

---

## 📁 Repository Structure

The project has been cleaned and structured for production deployment:

```text
Engage2Value/
├── app.py                      # Main Flask Web Server & API
├── custom_transformer.py       # ML Pipeline Scikit-Learn transformers
├── requirements.txt            # Python dependencies
│
├── data/                       # Contains the raw CSV and JSON samples
├── models/                     # Contains exported .joblib models & pipelines
├── notebooks/                  # Contains Jupyter notebooks & training scripts
│
├── static/                     # Frontend Javascript (main.js) & CSS
└── templates/                  # Frontend HTML Dashboard Structure
```

---

## 🚀 Installation & How to Run

1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Flask Application:**
   ```bash
   python app.py
   ```
4. **Access the Application:** Open your browser and navigate to `http://127.0.0.1:8000`.

---

## 👨‍💻 Author

**Tanush Sudheer Tambe**  
Final Year Engineering Student  
Specialization: IoT + Machine Learning + Data Science

### 📜 License
This project is released under the **MIT License** and is intended for educational and research purposes.
