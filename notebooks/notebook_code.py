# Basic Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# ML Libraries
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, silhouette_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor, StackingRegressor

from category_encoders import CountEncoder, TargetEncoder 

from lightgbm import LGBMClassifier, early_stopping, LGBMRegressor
from xgboost import XGBRegressor


# Suppressing warning messages
warnings.filterwarnings("ignore")

#Setting max rows to be shown to 300, showing all columns and with full width
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
print("Imports Done !")

df = pd.read_csv("/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv") 
X_test = pd.read_csv("/kaggle/input/engage-2-value-from-clicks-to-conversions/test_data.csv")

print("Train & Test data loaded !")

print(f"The shape of train data is: {df.shape}")
print(f"The shape of test data is: {X_test.shape}")

df.head()

def feature_summary(X, y):
    df = X
    records = []
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        count = len(series)
        nulls = series.isna().sum()
        null_pct = 100 * nulls/count
        uniques = series.nunique(dropna=False)
        try:
            corr = series.corr(y)
        except:
            corr = float('nan')
        records.append({
            'feature'    : col,
            'dtype'      : str(dtype),
            'count'      : count,
            'nulls'      : nulls,
            'null_pct'   : null_pct,
            'uniques'    : uniques,
            'corr_with_y': corr
        })  
    summary_df = pd.DataFrame(records)
    summary_df = summary_df[['feature','dtype','count','nulls',
                             'null_pct','uniques','corr_with_y']]
    return summary_df

print("Feature wise summary of train data:\n")
summary = feature_summary(df.drop(columns=["purchaseValue"]), df["purchaseValue"])
summary.sort_values(by = "corr_with_y", ascending=False)

print("Statistics for the numerical columns in dataset:")
df.describe().T

cat_features = ['trafficSource.medium', 'userChannel', 'locationCountry']
for feature in cat_features:
    print("\n")
    print("-"*40)
    print(df[feature].value_counts().head(10))
    print("-"*40)

missing_table = df.isnull().sum().reset_index()
missing_table.columns = ['feature', 'missing_count']
missing_table['missing_percent'] = (missing_table['missing_count'] / len(df)) * 100
missing_table = missing_table.sort_values('missing_percent', ascending=False)
missing_table[missing_table["missing_percent"]>0]

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid', palette='muted')
plt.figure(figsize=(8, 8))
sns.histplot(df['purchaseValue'], bins=20, kde=True)
plt.title('Distribution of PurchaseValue')
plt.show()

y = df['purchaseValue']
plt.figure(figsize=(20, 5))
sns.boxplot(x=y)
plt.title("Boxplot: Purchase Value")
plt.xlabel("purchaseValue")
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='whitegrid', palette='muted')
plt.figure(figsize=(8, 8))
sns.histplot(np.power((df['purchaseValue']), 1/ 1.54), bins=20, kde=True)
plt.title('Distribution of Power Transformed PurchaseValue')
plt.show()

y = np.power((df['purchaseValue']), 1/ 1.54)
plt.figure(figsize=(20, 5))
sns.boxplot(x=y)
plt.title("Boxplot: Purchase Value")
plt.xlabel("purchaseValue")
plt.tight_layout()
plt.show()

print("Original Target Skew: \t\t", df["purchaseValue"].skew())
print("Power Transformed Target Skew:\t",np.power((df['purchaseValue']), 1/ 1.54).skew())
print("log1p Transformed Target Skew:\t",np.log1p(df['purchaseValue']).skew())

num_features = ['purchaseValue', 'sessionNumber',
                'trafficSource.adwordsClickInfo.page',
                'pageViews', 'totalHits', ]
corr = df[num_features].corr()
plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap for Numerical Features')
plt.show()

plt.figure(figsize=(6, 6))
sns.scatterplot(x='pageViews', y='totalHits', data=df, palette='viridis', alpha=0.6)
plt.title('pageViews vs totalHits Scatter Plot')
plt.show()

plt.figure(figsize=(8, 7))
sns.countplot(y='browser', data=df, order=df['browser'].value_counts().index[:10])
plt.title('Top 10 Browsers Distribution')
plt.show()

for i in [df, X_test]:
    i['sessionStart_dt'] = pd.to_datetime(i['sessionStart'], unit='s', errors='coerce')
    i['session_dayofweek'] = i['sessionStart_dt'].dt.dayofweek
    i['session_hour'] = i['sessionStart_dt'].dt.hour
    i['session_month'] = i['sessionStart_dt'].dt.month
    i['is_weekend_session'] = i['session_dayofweek'].isin([5,6]).astype(float)
    i['is_business_hour'] = i['session_hour'].between(9, 17).astype(int)
    i['session_day'] = i['sessionStart_dt'].dt.day
    i['session_year'] = i['sessionStart_dt'].dt.year
    
    i['date'] = pd.to_datetime(i['date'], format='%Y%m%d')
    i['day'] = i['date'].dt.day
    i['month'] = i['date'].dt.month
    i['year'] = i['date'].dt.year
    i['day_of_week'] = i['date'].dt.dayofweek
    i['is_weekend'] = i['day_of_week'].isin([5, 6]).astype(float)
    i['quarter'] = i['date'].dt.quarter
    
print("Done !")

df_day = df.groupby('day_of_week')['purchaseValue'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x='day_of_week', y='purchaseValue', data=df_day, palette='viridis')
plt.title('Average Purchase by Day of Week')
plt.xlabel('Day of Week (0=Mon)')
plt.ylabel('Average Purchase')
plt.grid(True, axis='y')
plt.show()

df_date = df.groupby('date')['purchaseValue'].mean().reset_index()

plt.figure(figsize=(12, 5))
sns.lineplot(x='date', y='purchaseValue', data=df_date)
plt.title('Daily Average Purchase Over Time')
plt.xlabel('Date')
plt.ylabel('Avg Purchase')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

labels = ['Weekday', 'Weekend']
sizes = df['is_weekend'].value_counts(sort=False).values

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'orange'])
plt.title('Weekend vs Weekday Sessions')
plt.show()

df_q = df.groupby('quarter')['purchaseValue'].mean().reset_index()

plt.figure(figsize=(6, 4))
sns.barplot(x='quarter', y='purchaseValue', data=df_q, palette='viridis')
plt.title('Avg Purchase per Quarter')
plt.xlabel('Quarter')
plt.ylabel('Avg Purchase')
plt.show()

# Removing meaningless value from dataset
df = df.replace("not available in demo dataset", np.nan)
X_test = X_test.replace("not available in demo dataset", np.nan)

# df = df.dropna(axis=1, how='all')
# X_test = X_test.dropna(axis=1, how='all')
summary = feature_summary(df.drop(columns=["purchaseValue"]), df["purchaseValue"])
useless_cols = summary[
    ((summary['uniques'] == 1) 
     & (summary['nulls'] == 0)) 
     | (summary['null_pct'] == 100)
]['feature'].tolist()

print("Dropping useless columns:", useless_cols)
df = df.drop(columns=useless_cols)
X_test = X_test.drop(columns=useless_cols)

print(f"\nRemaining columns in train data: {df.shape[1]}\n")
summary = feature_summary(df.drop(columns=["purchaseValue"]), df["purchaseValue"])
summary.sort_values(by = "corr_with_y", ascending=False)

X = df.drop(columns='purchaseValue')
y = df['purchaseValue']

print("Shape of original data: ",X.shape, "\tShape of original target: ",y.shape)

y_strata = np.where(y == 0, 0, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_strata
)
print("Stratified Train Test Split DONE !")
print("\nShape after split->")
print("Train data:      ", X_train.shape, "\tTarget: ", y_train.shape)
print("Validation data: ", X_val.shape, "\tTarget: ", y_val.shape)

for df in [X_train, X_val, X_test]:
    df["trafficSource.isTrueDirect"] = df["trafficSource.isTrueDirect"].fillna(value=False)
    df["trafficSource.isTrueDirect"] = df["trafficSource.isTrueDirect"].astype(bool).astype(float)
    
    df["totals.bounces"] = df["totals.bounces"].fillna(value=0.0).astype(float)
    df["totals.bounces"]= df["totals.bounces"].astype(bool).astype(float)
    
    df["trafficSource.adwordsClickInfo.page"] = df["trafficSource.adwordsClickInfo.page"].fillna(0.0).astype(float)
    
    df["pageViews"] = df["pageViews"].fillna(0.0).astype(float)
    df["new_visits"] = df["new_visits"].fillna(0.0).astype(float)
    
    df["trafficSource.adwordsClickInfo.isVideoAd"] = df["trafficSource.adwordsClickInfo.isVideoAd"].fillna(1.0).astype(float)
    df["trafficSource.adwordsClickInfo.isVideoAd"] = df["trafficSource.adwordsClickInfo.isVideoAd"].astype(bool).astype(float)
    
    df["trafficSource.referralPath"] = df["trafficSource.referralPath"].fillna("Other")
    df["trafficSource.adContent"] = df["trafficSource.adContent"].fillna("Other")
    df["trafficSource.adwordsClickInfo.slot"] = df["trafficSource.adwordsClickInfo.slot"].fillna("Other")
    df["trafficSource.adwordsClickInfo.adNetworkType"] = df["trafficSource.adwordsClickInfo.adNetworkType"].fillna("Other")
    df["trafficSource.keyword"] = df["trafficSource.keyword"].fillna("(not provided)")
    df["geoNetwork.region"] = df["geoNetwork.region"].fillna("(not set)")
    df["geoNetwork.city"] = df["geoNetwork.city"].fillna("(not set)")
    df["geoNetwork.metro"] = df["geoNetwork.metro"].fillna("(not set)")
    
    df['browser'] = df['browser'].replace({
        'Firefox': 'Mozilla',
        'Mozilla Compatible Agent': 'Mozilla',
        'Mozilla': 'Mozilla',
        'Safari (in-app)': 'Safari'
    })
    
print("DONE !")

# -----------------------------------
# 1. Session & Interaction Features
# -----------------------------------
for df in [X_train, X_val, X_test]:
    df['hits_per_pageview'] = df['totalHits'] / (df['pageViews'] + 1)
    df['pageviews_per_hour'] = df['pageViews'] / (df['session_hour'] + 1)
    df['session_page_product'] = df['sessionNumber'] * df['pageViews']
    df['session_per_hit'] = df['sessionNumber'] / (df['totalHits'] + 1)
    df['bounce_hit_ratio'] = df['totals.bounces'] / (df['totalHits'] + 1)
    df['pageviews_per_channel'] = df['pageViews'] / (df['userChannel'].map(lambda x: 1 if x == 'Referral' else 2) + 1)

# -----------------------------------
# 2. Combined Features
# -----------------------------------
for df in [X_train, X_val, X_test]:
    df['is_repeat_visitor'] = (df['sessionNumber'] > 1).astype(float)
    df['is_video_ad_and_bounce'] = df['trafficSource.adwordsClickInfo.isVideoAd'] * df['totals.bounces']

# ---------------------------------------------
# 3. Frequency Encoding
# ---------------------------------------------
for col in ['browser', 'geoNetwork.city', 'trafficSource.campaign', 
            # 'userId', 'sessionId'
           ]:
    freqs = X_train[col].value_counts().to_dict()
    for df in [X_train, X_val, X_test]:
        df[f'{col}_freq'] = df[col].map(freqs).fillna(0)

# ---------------------------------------------
# 4. Time based Features
# ---------------------------------------------
for df in [X_train, X_val, X_test]:
    df['is_peak_month'] = df['month'].isin([4, 6, 8]).astype(int)
    df['is_off_season'] = df['month'].isin([10, 11]).astype(int)
    df['is_midweek'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
    df['is_Q2_or_Q3_peak'] = df['quarter'].isin([2, 3]).astype(int)
    df['is_peak_weekday'] = (df['day_of_week'] == 2).astype(int)


print("Added features!")

X_train_id = X_train.copy()
X_val_id = X_val.copy()
X_test_id = X_test.copy()

for col in ['userId', 'sessionId']:
    freqs = X_train_id[col].value_counts().to_dict()
    for df in [X_train_id, X_val_id, X_test_id]:
        df[f'{col}_freq'] = df[col].map(freqs).fillna(0)
        
for df in [X_train_id, X_val_id, X_test_id]:
    df.drop(columns=['sessionStart', 'sessionStart_dt', 'date'], inplace=True)
    df["userId"] = df["userId"].astype(str)
    df["sessionId"] = df["sessionId"].astype(str)

print("Columns from ID dataFrames dropped !")

for df in X_train, X_val, X_test:
    df.drop(columns=['sessionStart', 'sessionStart_dt', 
                     'date', "sessionId", "userId"], inplace=True)

print("Columns from leakage free dataFrame dropped !")

power = 1.54
best_power = 1.54

y_train_pow = np.power(y_train, 1 / power)
y_val_pow = np.power(y_val, 1/ power)

raw_num_cols = X_train.select_dtypes(include=['int32', 'int64', 'float32','float64']).columns.tolist()

NUM_COLS = [col for col in raw_num_cols]
CAT_COLS = list(set(X_train.columns) - set(NUM_COLS))
LOW_CARD = [col for col in CAT_COLS if X_train[col].nunique() <= 10]
HIGH_CARD = [col for col in CAT_COLS if X_train[col].nunique() > 10]

# ------------------------------Pipelines------------------------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

low_card_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

high_card_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder(smoothing=10))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, NUM_COLS),
    ('low_card', low_card_pipeline, LOW_CARD),
    ('high_card', high_card_pipeline, HIGH_CARD)
]).set_output(transform="pandas")

pipeline = Pipeline([
    ('preprocess', preprocessor)
])

# -------------------------------------------------------------------------
# Original target encoded data (NO IDs)
X_train_clean = pipeline.fit_transform(X_train, y_train)
X_val_clean = pipeline.transform(X_val)
X_test_clean = pipeline.transform(X_test)

# Power transform target encoded data (NO IDs)
X_train_clean_pow = pipeline.fit_transform(X_train, y_train_pow)
X_val_clean_pow = pipeline.transform(X_val)
X_test_clean_pow = pipeline.transform(X_test)

# -------------------------------------------------------------------------
# Original target encoded data (IDs INCLUDED)
X_train_clean_id = pipeline.fit_transform(X_train_id, y_train)
X_val_clean_id = pipeline.transform(X_val_id)
X_test_clean_id = pipeline.transform(X_test_id)

# Power transform target encoded data (IDs INCLUDED)
X_train_clean_pow_id = pipeline.fit_transform(X_train_id, y_train_pow)
X_val_clean_pow_id = pipeline.transform(X_val_id)
X_test_clean_pow_id = pipeline.transform(X_test_id)

print("Done !")

def add_post_pipeline_features(X_clean):
    X_clean['pageviews_business_hour_scaled'] = X_clean['num__pageViews'] * X_clean['num__is_business_hour']
    X_clean['hits_weekend_scaled'] = X_clean['num__totalHits'] * X_clean['num__is_weekend']
    X_clean['efficiency_business'] = X_clean['num__totalHits'] / (X_clean['num__is_business_hour'] + 1)
    X_clean['repeat_hits'] = X_clean['num__is_repeat_visitor'] * X_clean['num__totalHits']
    return X_clean

X_train_clean = add_post_pipeline_features(X_train_clean)
X_val_clean = add_post_pipeline_features(X_val_clean)
X_test_clean = add_post_pipeline_features(X_test_clean)

X_train_clean_pow = add_post_pipeline_features(X_train_clean_pow)
X_val_clean_pow = add_post_pipeline_features(X_val_clean_pow)
X_test_clean_pow = add_post_pipeline_features(X_test_clean_pow)

X_train_clean_pow_id = add_post_pipeline_features(X_train_clean_pow_id)
X_val_clean_pow_id = add_post_pipeline_features(X_val_clean_pow_id)
X_test_clean_pow_id = add_post_pipeline_features(X_test_clean_pow_id)

print("Post-pipeline features added!")

feature_summary(X_train_clean_pow, y_train_pow).sort_values(by="corr_with_y", ascending=False)

# ------------- Trimming original scale target encoded data -------------
fs = feature_summary(X_train_clean, y_train)
HIGH_CORR = fs[abs(fs["corr_with_y"])>0.01]["feature"].to_list()

X_train_trim = X_train_clean[HIGH_CORR]
X_val_trim = X_val_clean[HIGH_CORR]
X_test_trim = X_test_clean[HIGH_CORR]

# ------------- Trimming power transformed target encoded data -------------
fs = feature_summary(X_train_clean_pow, y_train_pow)
HIGH_CORR = fs[abs(fs["corr_with_y"])>0.01]["feature"].to_list()

X_train_trim_pow = X_train_clean_pow[HIGH_CORR]
X_val_trim_pow = X_val_clean_pow[HIGH_CORR]
X_test_trim_pow = X_test_clean_pow[HIGH_CORR]


# ------------- Trimming power transformed target encoded data containing IDs -------------
fs = feature_summary(X_train_clean_pow_id, y_train_pow)
HIGH_CORR = fs[abs(fs["corr_with_y"])>0.07]["feature"].to_list()

X_train_trim_pow_id = X_train_clean_pow_id[HIGH_CORR]
X_val_trim_pow_id = X_val_clean_pow_id[HIGH_CORR]
X_test_trim_pow_id = X_test_clean_pow_id[HIGH_CORR]

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_trim_pow)

# inertias = []
# silhouette_scores = []
# k_values = range(2, 11)

# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_train_scaled)
    
#     inertia = kmeans.inertia_
#     silhouette = silhouette_score(X_train_scaled, labels)

#     inertias.append(inertia)
#     silhouette_scores.append(silhouette)

# print(np.round(inertias,2),"\n")
# print(np.round(silhouette_scores, 4))

# '''
# [6580022.73 6114629.99 5757691.67 5545888.83 5280742.17 5095057.5
#  4838553.64 4747861.73 4632441.6 ] 
##[2      3      4      5      6      7      8      9      10    ]
# [0.1233 0.1372 0.1117 0.0833 0.0989 0.0972 0.1025 0.0955 0.0936]
# '''

scaler = StandardScaler()
X_train_trim_scaled = scaler.fit_transform(X_train_trim_pow)
X_val_trim_scaled = scaler.transform(X_val_trim_pow)
X_test_trim_scaled = scaler.transform(X_test_trim_pow)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_train_trim_scaled)

X_train_stack = X_train_trim_pow.copy()
X_val_stack = X_val_trim_pow.copy()
X_test_stack = X_test_trim_pow.copy()

X_train_stack['cluster'] = kmeans.labels_
X_val_stack['cluster'] = kmeans.predict(X_val_trim_scaled)
X_test_stack['cluster'] = kmeans.predict(X_test_trim_scaled)

print("Clustering added! New 'cluster' feature created.")

X_full = pd.concat([X_train_stack, X_val_stack])
y_full_pow = pd.concat([y_train_pow, y_val_pow])

# Binary target for classifier
y_full_bin = (y_full_pow> 0).astype(int)
y_train_bin = (y_train_pow > 0).astype(int)
y_val_bin = (y_val_pow > 0).astype(int)


clf = LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=8,
    num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1
)

clf.fit(X_train_stack, y_train_bin)
val_bin_pred = clf.predict(X_val_stack)

acc = accuracy_score(y_val_bin, val_bin_pred)
prec = precision_score(y_val_bin, val_bin_pred)
rec = recall_score(y_val_bin, val_bin_pred)
f1 = f1_score(y_val_bin, val_bin_pred)

print(f"Classifier Validation Accuracy: {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall: {rec:.4f}")
print(f"Validation F1 Score: {f1:.4f}")

clf.fit(X_full, y_full_bin)
train_probs = clf.predict_proba(X_train_stack)[:, 1]
val_probs = clf.predict_proba(X_val_stack)[:, 1]
test_proba = clf.predict_proba(X_test_stack)[:, 1]

X_train_stack['buy_prob'] = train_probs
X_val_stack['buy_prob'] = val_probs
X_test_stack['buy_prob'] = test_proba

feature_summary(X_train_stack, y_train_pow).sort_values(by="corr_with_y", ascending=False)

lgb_model = LGBMRegressor(random_state=42, verbosity=-1)

# lgb_model.fit(X_train_stack, y_train_pow)

# y_train_pred_pow = lgb_model.predict(X_train_stack)
# y_train_pred = np.power(np.maximum(y_train_pred_pow, 0), power)

# r2_train_lgb = r2_score(y_train, y_train_pred)
# print(f"LightGBM R2 on Train: {r2_train_lgb:.4f}")

# y_val_pred_pow = lgb_model.predict(X_val_stack)
# y_val_pred = np.power(np.maximum(y_val_pred_pow, 0), power)

# r2_val_lgb = r2_score(y_val, y_val_pred)
# print(f"LightGBM R2 on Validation: {r2_val_lgb:.4f}")

xgb_model = XGBRegressor(random_state=42, verbosity=0)

# xgb_model.fit(X_train_stack, y_train_pow)

# y_train_pred_pow = xgb_model.predict(X_train_stack)
# y_train_pred = np.power(np.maximum(y_train_pred_pow, 0), power)

# r2_train_xgb = r2_score(y_train, y_train_pred)
# print(f"XGBoost R2 on Train: {r2_train_xgb:.4f}")

# y_val_pred_pow = xgb_model.predict(X_val_stack)
# y_val_pred = np.power(np.maximum(y_val_pred_pow, 0), power)

# r2_val_xgb = r2_score(y_val, y_val_pred)
# print(f"XGBoost R2 on Validation: {r2_val_xgb:.4f}")

et_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)

# et_model.fit(X_train_stack, y_train_pow)

# y_train_pred_pow = et_model.predict(X_train_stack)
# y_train_pred = np.power(np.maximum(y_train_pred_pow, 0), power)

# r2_train_et = r2_score(y_train, y_train_pred)
# print(f"ExtraTrees R2 on Train: {r2_train_et:.4f}")

# y_val_pred_pow = et_model.predict(X_val_stack)
# y_val_pred = np.power(np.maximum(y_val_pred_pow, 0), power)

# r2_val_et = r2_score(y_val, y_val_pred)
# print(f"ExtraTrees R2 on Validation: {r2_val_et:.4f}")

# models = ['LightGBM', 'XGBoost ', 'ExtraTrees']
# train_r2 = [r2_train_lgb, r2_train_xgb, r2_train_et]
# val_r2 = [r2_val_lgb, r2_val_xgb, r2_val_et]
# print("Model\t\t\tTrain R2\tValidation R2")
# for i in range(len(models)):
#     print(f"{models[i]}\t\t{train_r2[i]:.4f}\t\t{val_r2[i]:.4f}")
# x = range(len(models))

# plt.figure(figsize=(9,5))
# plt.bar(x, train_r2, width=0.4, label='Train R2', align='center')
# plt.bar([i + 0.4 for i in x], val_r2, width=0.4, label='Validation R2', align='center')
# plt.xticks([i + 0.2 for i in x], models, fontsize=7)
# plt.ylabel("R2 Score")
# plt.title("Model Comparison")
# plt.legend(loc='upper right', fontsize=7)
# plt.show()

power = 1.54

xgb_param_grid = {
    'n_estimators': [900, 1000, 1100, 1300],
    'learning_rate': [0.03, 0.04, 0.05],
    'max_depth': [8, 10, 12, 15],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0, 0.5, 1, 5],
    'reg_alpha': [0, 5, 10, 15],
    'reg_lambda': [0, 5, 10, 15]
}

xgb_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    # verbosity=0,
    tree_method='hist'
)

xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_grid,
    n_iter=20,
    scoring='r2',
    cv=3,
    verbose=1,
    random_state=47,
    n_jobs=-1
)

# xgb_search.fit(X_train_stack, np.power(y_train, 1/power))

# print("Best Hyperparameters:\n", xgb_search.best_params_)
# print("Best CV R² Score:", round(xgb_search.best_score_, 4))

# best_xgb = xgb_search.best_estimator_

# y_train_pred_pow = best_xgb.predict(X_train_stack)
# y_train_pred = np.power(np.maximum(y_train_pred_pow, 0), power)

# y_val_pred_pow = best_xgb.predict(X_val_stack)
# y_val_pred = np.power(np.maximum(y_val_pred_pow, 0), power)

# r2_train = r2_score(y_train, y_train_pred)
# r2_val = r2_score(y_val, y_val_pred)

# print(f"Train R2: {r2_train:.4f} | Validation R2: {r2_val:.4f}")

xgb_best_params = {
    'subsample': 0.8, 
    'reg_lambda': 10, 
    'reg_alpha': 10, 
    'n_estimators': 1000, 
    'max_depth': 12, 
    'learning_rate': 0.04, 
    'gamma': 1, 
    'colsample_bytree': 0.7
}

xgb_model=XGBRegressor(
    random_state = 42, verbosity=0,
    **xgb_best_params
)


# xgb_model.fit(X_train_stack, y_train_pow)

# y_train_pred_pow = xgb_model.predict(X_train_stack)
# y_train_pred = np.power(np.maximum(y_train_pred_pow, 0), power)

# y_val_pred_pow = xgb_model.predict(X_val_stack)
# y_val_pred = np.power(np.maximum(y_val_pred_pow, 0), power)

# r2_train_xgb = r2_score(y_train, y_train_pred)
# r2_val_xgb = r2_score(y_val, y_val_pred)
# print(f"R2 on Train: {r2_train_xgb:.4f} | R2 on Validation: {r2_val_xgb:.4f}")

param_grid = {
    'n_estimators': [900, 1000, 1100, 1300],
    'learning_rate': [0.03, 0.04, 0.05],
    'max_depth': [12, 15, 17, 20, 22],
    'num_leaves': [256, 384, 416, 512, 768],
    'min_child_samples': [4, 5, 7, 9],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [15, 18, 20, 25, 35],
    'reg_lambda': [15, 18, 20, 25, 35],
}

lgb_model = LGBMRegressor(random_state=24, verbosity=-1)

search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    n_iter=50,
    scoring='r2',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

# search.fit(X_train_stack, y_train_pow)

# print("Best Hyperparameters:\n", search.best_params_)
# print("Best CV R2 Score:", round(search.best_score_,4))

# best_params= search.best_params_
# model = search.best_estimator_

# y_train_pred_pow = model.predict(X_train_stack)
# y_train_pred = np.power(np.maximum(y_train_pred_pow, 0), power)

# y_val_pred_pow = model.predict(X_val_stack)
# y_val_pred = np.power(np.maximum(y_val_pred_pow, 0), power)

# r2_train = r2_score(y_train, y_train_pred)
# r2_val = r2_score(y_val, y_val_pred)
# print(f"R2 on Train: {r2_train:.4f} | R2 on Validation: {r2_val:.4f}")

best_params={
    'subsample': 0.9, 
    'reg_lambda': 15, 
    'reg_alpha': 20, 
    'num_leaves': 512, 
    'n_estimators': 1000, 
    'min_child_samples': 5, 
    'max_depth': 20, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.6
}

best_model = None

lgbm_model=LGBMRegressor(random_state = 42, verbosity = -1, **best_params)

# lgbm_model.fit(X_train_stack, y_train_pow)

# y_train_pred_pow = lgbm_model.predict(X_train_stack)
# y_train_pred = np.power(np.maximum(y_train_pred_pow, 0), power)

# y_val_pred_pow = lgbm_model.predict(X_val_stack)
# y_val_pred = np.power(np.maximum(y_val_pred_pow, 0), power)

# r2_train_lgb = r2_score(y_train, y_train_pred)
# r2_val_lgb = r2_score(y_val, y_val_pred)
# print(f"R2 on Train: {r2_train_lgb:.4f} | R2 on Validation: {r2_val_lgb:.4f} gap: {(r2_train_lgb-r2_val_lgb):.4f}")

# best_model = lgbm_model

best_params_reg={
    'subsample': 0.9, 
    'reg_lambda': 20, 
    'reg_alpha': 20,
    'num_leaves': 512, 
    'n_estimators': 1000, 
    'min_child_samples': 5, 
    'max_depth': 20, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.5
}

best_model = None

lgbm_model_id=LGBMRegressor(random_state = 42, verbosity = -1, **best_params_reg)

lgbm_model_id.fit(X_train_trim_pow_id, y_train_pow)

y_train_pred_pow_id = lgbm_model_id.predict(X_train_trim_pow_id)
y_train_pred_id = np.power(np.maximum(y_train_pred_pow_id, 0), power)

y_val_pred_pow_id = lgbm_model_id.predict(X_val_trim_pow_id)
y_val_pred_id = np.power(np.maximum(y_val_pred_pow_id, 0), power)

r2_train_lgb_id = r2_score(y_train, y_train_pred_id)
r2_val_lgb_id = r2_score(y_val, y_val_pred_id)
print(f"R2 on Train: {r2_train_lgb_id:.4f} | R2 on Validation: {r2_val_lgb_id:.4f} gap: {(r2_train_lgb_id-r2_val_lgb_id):.4f}")

best_model_id = lgbm_model_id

best_params_1={
    'subsample': 0.9, 
    'reg_lambda': 15, 
    'reg_alpha': 20, 
    'num_leaves': 512, 
    'n_estimators': 1000, 
    'min_child_samples': 5, 
    'max_depth': 20, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.6
}
best_params_2={
    'subsample': 0.9, 
    'reg_lambda': 13, 
    'reg_alpha': 18, 
    'num_leaves': 512, 
    'n_estimators': 1000, 
    'min_child_samples': 5, 
    'max_depth': 20, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.6
}
best_params_3={
    'subsample': 0.9, 
    'reg_lambda': 17, 
    'reg_alpha': 22, 
    'num_leaves': 512, 
    'n_estimators': 1000, 
    'min_child_samples': 5, 
    'max_depth': 20, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.6
}
best_params_4={
    'subsample': 0.9, 
    'reg_lambda': 15, 
    'reg_alpha': 20, 
    'num_leaves': 512, 
    'n_estimators': 1300, 
    'min_child_samples': 5, 
    'max_depth': 20, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.6
}
best_params_5={
    'subsample': 0.9, 
    'reg_lambda': 15, 
    'reg_alpha': 20, 
    'num_leaves': 512, 
    'n_estimators': 1300, 
    'min_child_samples': 5, 
    'max_depth': 16, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.6
}

power = 1.54
ensemble_2 = VotingRegressor(
    estimators=[('lgb1', LGBMRegressor(random_state = 42, verbosity = -1, **best_params_1)),
                ('lgb2', LGBMRegressor(random_state = 41, verbosity = -1, **best_params_2)),
                ('lgb3', LGBMRegressor(random_state = 43, verbosity = -1, **best_params_3)),
                ('lgb4', LGBMRegressor(random_state = 40, verbosity = -1, **best_params_4)),
                ('lgb5', LGBMRegressor(random_state = 47, verbosity = -1, **best_params_5)),
               ],      
)


# ensemble_2.fit(X_train_stack, y_train_pow)

# train_preds_pow = ensemble_2.predict(X_train_stack)
# val_preds_pow = ensemble_2.predict(X_val_stack)

# train_preds = np.power(np.maximum(train_preds_pow, 0), power)
# val_preds = np.power(np.maximum(val_preds_pow, 0), power)

# r2_train_es = r2_score(y_train, train_preds)
# r2_val_es = r2_score(y_val, val_preds)

# best_model = ensemble_2
# print(f"Ensemble R2 on Train: {r2_train_es:.4f}")
# print(f"Ensemble R2 on Validation: {r2_val_es:.4f}")

# models = ['Tuned LightGBM No IDs', 'Tuned LightGBM with IDs', 'Tuned XGBoost No IDs','Tuned Ensemble No IDs']
# train_r2 = [r2_train_lgb,r2_train_lgb_id, r2_train_xgb, r2_train_es]
# val_r2 = [r2_val_lgb,r2_val_lgb_id, r2_val_xgb, r2_val_es]
# print("Model\t\t\t\tTrain R2\tValidation R2")
# for i in range(len(models)):
#     print(f"{models[i]}\t\t{train_r2[i]:.4f}\t\t{val_r2[i]:.4f}")

# x = range(len(models))

# plt.figure(figsize=(9,5))
# plt.bar(x, train_r2, width=0.4, label='Train R2', align='center')
# plt.bar([i + 0.4 for i in x], val_r2, width=0.4, label='Validation R2', align='center')

# plt.xticks([i + 0.2 for i in x], models, fontsize=6)
# plt.ylabel("R2 Score")
# plt.ylim(0.7, 1.0)
# plt.title("Model Comparison")
# plt.legend(loc='upper right', fontsize=7)
# plt.show()

best_model = lgbm_model_id


X_full = pd.concat([X_train_stack, X_val_stack])
y_full_pow = pd.concat([y_train_pow, y_val_pow])
X_full_id = pd.concat([X_train_trim_pow_id, X_val_trim_pow_id])

best_model.fit(X_full_id, y_full_pow)
print("Model Fitted !")

X_full.shape, X_train_stack.shape

best_power=1.54

# y_test_pred_pow = best_model_1.predict(X_test_stack)
# y_test_pred = np.power(np.maximum(y_test_pred_pow, 0), best_power)

y_test_pred_pow_id = best_model.predict(X_test_trim_pow_id)
y_test_pred = np.power(np.maximum(y_test_pred_pow_id, 0), best_power)

# y_test_pred_hybrid = np.where(y_test_pred_id >1000, y_test_pred_id, y_test_pred)



min_nonzero = y_train[y_train > 0].min()
clip_threshold = min_nonzero / 1.5

y_pred_clipped = y_test_pred.copy()
y_pred_clipped[y_pred_clipped < clip_threshold] = 0
y_pred = y_pred_clipped
print("Prediction Done !")

y_pred.shape[0], X_test_clean.shape[0]

submission = pd.DataFrame({"id": range(0,X_test.shape[0]), "purchaseValue": y_pred}) 
submission.to_csv('submission.csv',index=False)
print("DONE")