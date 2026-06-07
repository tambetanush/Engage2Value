import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier, LGBMRegressor
import sys
import warnings
warnings.filterwarnings("ignore")

print("1. Loading pipeline and data...")
# Load custom module into scope
import custom_transformer
sys.modules['__main__'].CustomFeatureEngineer = custom_transformer.CustomFeatureEngineer

data_pipeline = joblib.load('full_data_pipeline.joblib')

df_train = pd.read_csv('train_data.csv')
y_train = df_train['purchaseValue']
power = 1.54
y_train_pow = np.power(y_train, 1 / power)

print("2. Processing data...")
X_train_clean = data_pipeline.transform(df_train)

def add_post_pipeline_features(X_clean):
    X_clean['pageviews_business_hour_scaled'] = X_clean['num__pageViews'] * X_clean['num__is_business_hour']
    X_clean['hits_weekend_scaled'] = X_clean['num__totalHits'] * X_clean['num__is_weekend']
    X_clean['efficiency_business'] = X_clean['num__totalHits'] / (X_clean['num__is_business_hour'] + 1)
    X_clean['repeat_hits'] = X_clean['num__is_repeat_visitor'] * X_clean['num__totalHits']
    return X_clean

X_train_clean_pow = add_post_pipeline_features(X_train_clean)

print("3. Calculating HIGH_CORR features...")
corr_scores = {}
for col in X_train_clean_pow.columns:
    try:
        corr_scores[col] = X_train_clean_pow[col].corr(y_train_pow)
    except:
        pass

HIGH_CORR = [col for col, corr in corr_scores.items() if abs(corr) > 0.01]
print(f"Number of HIGH_CORR features: {len(HIGH_CORR)}")
joblib.dump(HIGH_CORR, 'high_corr_features.joblib')

X_train_trim_pow = X_train_clean_pow[HIGH_CORR]

print("4. Training Scaler & KMeans...")
scaler = StandardScaler()
X_train_trim_scaled = scaler.fit_transform(X_train_trim_pow)
joblib.dump(scaler, 'scaler_trim.joblib')

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_train_trim_scaled)
joblib.dump(kmeans, 'kmeans_model.joblib')

X_train_stack = X_train_trim_pow.copy()
X_train_stack['cluster'] = kmeans.labels_

print("5. Training LGBMClassifier...")
y_train_bin = (y_train_pow > 0).astype(int)
clf = LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=8, num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1)
clf.fit(X_train_stack, y_train_bin)
joblib.dump(clf, 'lgbm_classifier.joblib')

train_probs = clf.predict_proba(X_train_stack)[:, 1]
X_train_stack['buy_prob'] = train_probs

print("6. Training Final Regressor...")
best_params_reg = {
    'subsample': 0.9, 'reg_lambda': 20, 'reg_alpha': 20,
    'num_leaves': 512, 'n_estimators': 1000, 'min_child_samples': 5, 
    'max_depth': 20, 'learning_rate': 0.05, 'colsample_bytree': 0.5
}
final_model = LGBMRegressor(random_state=42, verbosity=-1, **best_params_reg)
final_model.fit(X_train_stack, y_train_pow)
joblib.dump(final_model, 'stacked_lgbm_regressor.joblib')

print("SUCCESS: All models retrained and saved.")
