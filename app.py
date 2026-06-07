import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import shap
from scipy import stats
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import IsolationForest

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask App
app = Flask(__name__)

# --- Model & Pipeline Loading ---
POWER_FACTOR = 1.54
PIPELINE_PATH = "models/full_data_pipeline.joblib"

# The Custom Transformer logic
try:
    from custom_transformer import CustomFeatureEngineer
    sys.modules['custom_transformer'] = sys.modules['__main__']
    data_pipeline = joblib.load(PIPELINE_PATH)
    
    # Load stacked models
    high_corr_features = joblib.load('models/high_corr_features.joblib')
    scaler_trim = joblib.load('models/scaler_trim.joblib')
    kmeans_model = joblib.load('models/kmeans_model.joblib')
    lgbm_classifier = joblib.load('models/lgbm_classifier.joblib')
    model = joblib.load('models/stacked_lgbm_regressor.joblib')
    
    logger.info("✅ Pipeline and models loaded successfully!")
except Exception as e:
    logger.error(f"🚨 Error loading pipeline/models: {e}")
    data_pipeline = None
    model = None

# --- Data Preloading ---
DATA_PATH = "data/train_data.csv"
try:
    logger.info("Loading dataset for exploration...")
    # Load a sample to keep things fast, or full if memory allows. 50k is a good balance.
    df_train = pd.read_csv(DATA_PATH, nrows=50000)
    
    # Calculate frequency maps for custom_transformer
    freq_maps = {}
    for col in ['browser', 'geoNetwork.city', 'trafficSource.campaign']:
        if col in df_train.columns:
            freq_maps[col] = df_train[col].value_counts().to_dict()
            
    # Clean column names just like the pipeline expects, and set target
    TARGET_COL = 'purchaseValue'
    logger.info("✅ Dataset loaded!")
except Exception as e:
    logger.error(f"🚨 Error loading dataset: {e}")
    df_train = pd.DataFrame()
    freq_maps = {}


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/features")
def get_features():
    if df_train.empty:
        return jsonify({"error": "Dataset not loaded."}), 500
    
    features = [c for c in df_train.columns if c != TARGET_COL]
    return jsonify({"features": sorted(features)})


@app.route("/api/explore/<feature_name>")
def explore_feature(feature_name):
    if df_train.empty or feature_name not in df_train.columns:
        return jsonify({"error": "Feature not found."}), 404

    target = df_train[TARGET_COL].fillna(0)
    feature_series = df_train[feature_name]
    
    is_numeric = pd.api.types.is_numeric_dtype(feature_series)
    
    # 1. Basic Stats
    missing_pct = float(feature_series.isnull().mean() * 100)
    unique_vals = int(feature_series.nunique())
    
    stats_dict = {
        "missing_pct": round(missing_pct, 2),
        "unique_vals": unique_vals,
        "type": "Numeric" if is_numeric else "Categorical"
    }

    plots_data = {}
    inference = []

    # Drop NA for calculations
    valid_idx = feature_series.dropna().index
    valid_f = feature_series.loc[valid_idx]
    valid_t = target.loc[valid_idx]

    if is_numeric:
        stats_dict["mean"] = float(valid_f.mean()) if not pd.isna(valid_f.mean()) else 0.0
        stats_dict["median"] = float(valid_f.median()) if not pd.isna(valid_f.median()) else 0.0
        stats_dict["min"] = float(valid_f.min()) if not pd.isna(valid_f.min()) else 0.0
        stats_dict["max"] = float(valid_f.max()) if not pd.isna(valid_f.max()) else 0.0
        
        # Distribution (Histogram data)
        hist, bins = np.histogram(valid_f, bins=20)
        plots_data['distribution'] = {
            "x": [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)],
            "y": hist.tolist(),
            "type": "bar",
            "name": "Distribution"
        }
        
        # Target Comparison (Scatter sample)
        sample_size = min(1000, len(valid_f))
        sample_idx = np.random.choice(valid_idx, sample_size, replace=False)
        
        sample_f = valid_f.loc[sample_idx].astype(float)
        sample_t = valid_t.loc[sample_idx].astype(float)
        
        # --- NEW: Anomaly Detection ---
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        X_iso = sample_f.values.reshape(-1, 1)
        anomaly_preds = iso_forest.fit_predict(X_iso)
        is_anomaly = (anomaly_preds == -1).tolist()

        plots_data['target_scatter'] = {
            "x": sample_f.tolist(),
            "y": sample_t.tolist(),
            "is_anomaly": is_anomaly,
            "type": "scatter",
            "mode": "markers",
            "name": "Feature vs Target"
        }

        # Correlation
        if valid_f.std() > 0:
            corr, p_val = stats.pearsonr(valid_f, valid_t)
            stats_dict["correlation"] = round(corr, 4)
            if abs(corr) > 0.3:
                inference.append(f"There is a significant {'positive' if corr > 0 else 'negative'} correlation ({corr:.2f}) with {TARGET_COL}.")
            else:
                inference.append(f"There is a weak/no linear correlation ({corr:.2f}) with {TARGET_COL}.")
        else:
            stats_dict["correlation"] = 0
            inference.append("Feature has zero variance, no correlation can be found.")

        # Empirical PDP
        try:
            bins, bin_edges = pd.qcut(valid_f, q=10, retbins=True, duplicates='drop')
            pdp_means = valid_t.groupby(bins, observed=False).mean()
            labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
            if len(labels) == len(pdp_means):
                plots_data['pdp'] = {
                    "x": labels,
                    "y": pdp_means.values.tolist(),
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Marginal Effect (PDP)",
                    "line": {"color": "#8b5cf6", "width": 3},
                    "marker": {"size": 8}
                }
        except Exception as e:
            logger.error(f"PDP error: {e}")

        # Subgroup Discovery
        try:
            from sklearn.tree import DecisionTreeRegressor
            dt_sub = DecisionTreeRegressor(max_depth=2, min_samples_leaf=0.05, random_state=42)
            X_sub = valid_f.values.reshape(-1, 1)
            dt_sub.fit(X_sub, valid_t)
            
            tree = dt_sub.tree_
            leaves = [i for i in range(tree.node_count) if tree.children_left[i] == -1]
            if leaves:
                best_leaf = max(leaves, key=lambda i: tree.value[i][0][0])
                best_val = tree.value[best_leaf][0][0]
                
                def find_path(curr, target, current_path):
                    if curr == target: return current_path
                    if tree.children_left[curr] == -1: return None
                    l = find_path(tree.children_left[curr], target, current_path + [(curr, "<=")])
                    if l: return l
                    return find_path(tree.children_right[curr], target, current_path + [(curr, ">")])
                    
                path = find_path(0, best_leaf, [])
                if path:
                    conditions = [f"{op} {tree.threshold[n]:.2f}" for n, op in path]
                    cond_str = " AND ".join(conditions)
                    baseline = valid_t.mean()
                    if baseline > 0 and best_val > baseline * 1.2:
                        pct = ((best_val - baseline)/baseline)*100
                        stats_dict['subgroup_insight'] = f"Users with {feature_name} {cond_str} have an average purchase value of {best_val:.2f} (+{pct:.0f}% above average)."
        except Exception as e:
            logger.error(f"Subgroup error: {e}")

    else:
        # Categorical
        val_counts = valid_f.value_counts().head(10)
        stats_dict["mode"] = str(val_counts.index[0]) if len(val_counts) > 0 else "N/A"
        
        plots_data['distribution'] = {
            "x": val_counts.index.astype(str).tolist(),
            "y": val_counts.values.astype(int).tolist(),
            "type": "bar",
            "name": "Top Categories"
        }
        
        # Target Comparison (Boxplot data proxy via means for simplicity)
        means = valid_t.groupby(valid_f).mean().sort_values(ascending=False).head(10)
        plots_data['target_scatter'] = {
            "x": means.index.astype(str).tolist(),
            "y": means.values.astype(float).tolist(),
            "type": "bar",
            "name": "Mean Target by Category"
        }
        
        inference.append(f"The most common category is '{stats_dict['mode']}'.")
        if len(means) > 0:
            inference.append(f"Category '{means.index[0]}' has the highest average {TARGET_COL}.")

    if missing_pct > 20:
        inference.append("Warning: High percentage of missing values. Imputation or dropping might be necessary.")

    return jsonify({
        "stats": stats_dict,
        "plots": plots_data,
        "inference": " ".join(inference)
    })

@app.route("/api/bivariate/<f1>/<f2>")
def bivariate(f1, f2):
    if df_train.empty or f1 not in df_train.columns or f2 not in df_train.columns:
        return jsonify({"error": "Invalid features"}), 400
        
    df = df_train[[f1, f2, TARGET_COL]].dropna()
    is_num1 = pd.api.types.is_numeric_dtype(df[f1])
    is_num2 = pd.api.types.is_numeric_dtype(df[f2])
    
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
        
    res = {"type": None, "plot": {}}
    
    if is_num1 and is_num2:
        res["type"] = "scatter"
        res["plot"] = {
            "x": df[f1].tolist(),
            "y": df[f2].tolist(),
            "marker": {
                "color": df[TARGET_COL].tolist(),
                "colorscale": "Roses",
                "showscale": True,
                "colorbar": {"title": "Target"},
                "size": 8,
                "opacity": 0.8
            },
            "mode": "markers",
            "type": "scatter",
            "name": "Bivariate"
        }
    elif not is_num1 and not is_num2:
        res["type"] = "heatmap"
        top_f1 = df[f1].value_counts().head(5).index
        top_f2 = df[f2].value_counts().head(5).index
        df_sub = df[df[f1].isin(top_f1) & df[f2].isin(top_f2)]
        pivot = df_sub.pivot_table(values=TARGET_COL, index=f1, columns=f2, aggfunc='mean').fillna(0)
        res["plot"] = {
            "z": pivot.values.tolist(),
            "x": pivot.columns.astype(str).tolist(),
            "y": pivot.index.astype(str).tolist(),
            "type": "heatmap",
            "colorscale": "Roses"
        }
    else:
        res["type"] = "boxplot"
        num_f, cat_f = (f1, f2) if is_num1 else (f2, f1)
        top_cat = df[cat_f].value_counts().head(5).index
        df_sub = df[df[cat_f].isin(top_cat)]
        
        traces = []
        for cat in top_cat:
            traces.append({
                "y": df_sub[df_sub[cat_f] == cat][num_f].tolist(),
                "type": "box",
                "name": str(cat)
            })
        res["plot"] = traces
        res["num_f"] = num_f
        res["cat_f"] = cat_f

    return jsonify(res)

@app.route("/api/shap-summary")
def get_shap_summary():
    if not data_pipeline or not model or df_train.empty:
        return jsonify({"error": "Models or data not loaded."}), 503
        
    try:
        sample_size = min(1, len(df_train))
        df_sample = df_train.sample(n=sample_size, random_state=42).copy()
        
        expected_cols = [c for c in df_train.columns if c != TARGET_COL]
        df_input = df_sample[expected_cols]
        
        for col in expected_cols:
            if pd.api.types.is_numeric_dtype(df_train[col]):
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
                
        useless_cols = [
            'device.screenResolution', 'device.mobileDeviceBranding',
            'device.mobileInputSelector', 'device.mobileDeviceMarketingName',
            'device.operatingSystemVersion', 'device.flashVersion',
            'geoNetwork.networkLocation', 'browserMajor', 'device.browserSize',
            'device.mobileDeviceModel', 'device.language', 'device.browserVersion',
            'device.screenColors'
        ]
        for col in useless_cols:
            if col in df_input.columns:
                df_input[col] = np.nan
                
        X_processed = data_pipeline.transform(df_input)
        X_processed['pageviews_business_hour_scaled'] = X_processed['num__pageViews'] * X_processed['num__is_business_hour']
        X_processed['hits_weekend_scaled'] = X_processed['num__totalHits'] * X_processed['num__is_weekend']
        X_processed['efficiency_business'] = X_processed['num__totalHits'] / (X_processed['num__is_business_hour'] + 1)
        X_processed['repeat_hits'] = X_processed['num__is_repeat_visitor'] * X_processed['num__totalHits']
        
        X_trim = X_processed[high_corr_features]
        X_stack = X_trim.copy()
        X_stack['cluster'] = 0
        X_stack['buy_prob'] = 0.0
        
        # Native Feature Importances instead of SHAP for instant load
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(X_stack.columns))
            
        feature_names = X_stack.columns.tolist()
        shap_summary = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        
        top_features = [x[0] for x in shap_summary]
        top_importances = [float(x[1]) for x in shap_summary]
        
        return jsonify({
            "features": top_features[::-1],
            "importances": top_importances[::-1]
        })
        
    except Exception as e:
        logger.error(f"SHAP Summary Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    if not data_pipeline or not model:
        return jsonify({"error": "Model or pipeline not loaded."}), 503
        
    try:
        data = request.json
        df_input = pd.DataFrame([data])
        # Replace underscores with dots as per original logic, except for new_visits
        df_input.columns = [col.replace("_", ".") if col != 'new_visits' else col for col in df_input.columns]
        
        # Datatype fixes
        if 'userId' in df_input.columns: df_input['userId'] = df_input['userId'].astype(str)
        if 'sessionId' in df_input.columns: df_input['sessionId'] = df_input['sessionId'].astype(str)
        
        # Ensure column order matches the training data (excluding target)
        if not df_train.empty:
            expected_cols = [c for c in df_train.columns if c != TARGET_COL]
            for c in expected_cols:
                if c not in df_input.columns:
                    df_input[c] = np.nan
            df_input = df_input[expected_cols]
            
            # Coerce expected numeric columns to numbers, turning invalid strings into NaN
            for col in expected_cols:
                if pd.api.types.is_numeric_dtype(df_train[col]):
                    df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
                    
            # Force columns that were 100% missing during training to NaN.
            # In the pipeline, they are mistakenly treated as numeric because replacing 
            # "not available..." made them 100% NaN (float) during pipeline fit.
            useless_cols = [
                'device.screenResolution', 'device.mobileDeviceBranding',
                'device.mobileInputSelector', 'device.mobileDeviceMarketingName',
                'device.operatingSystemVersion', 'device.flashVersion',
                'geoNetwork.networkLocation', 'browserMajor', 'device.browserSize',
                'device.mobileDeviceModel', 'device.language', 'device.browserVersion',
                'device.screenColors'
            ]
            for col in useless_cols:
                if col in df_input.columns:
                    df_input[col] = np.nan
            
        # 1. Pipeline
        X_processed = data_pipeline.transform(df_input)
        
        # 2. Post-pipeline features
        X_processed['pageviews_business_hour_scaled'] = X_processed['num__pageViews'] * X_processed['num__is_business_hour']
        X_processed['hits_weekend_scaled'] = X_processed['num__totalHits'] * X_processed['num__is_weekend']
        X_processed['efficiency_business'] = X_processed['num__totalHits'] / (X_processed['num__is_business_hour'] + 1)
        X_processed['repeat_hits'] = X_processed['num__is_repeat_visitor'] * X_processed['num__totalHits']
        
        # 3. Trim to high correlation features
        X_trim = X_processed[high_corr_features]
        
        # 4. Scale and Cluster
        X_scaled = scaler_trim.transform(X_trim)
        cluster = kmeans_model.predict(X_scaled)
        
        # 5. Stacking
        X_stack = X_trim.copy()
        X_stack['cluster'] = cluster
        
        buy_prob = lgbm_classifier.predict_proba(X_stack)[:, 1]
        X_stack['buy_prob'] = buy_prob
        
        # 6. Final Prediction
        prediction_pow = model.predict(X_stack)
        final_prediction = np.power(np.maximum(0, prediction_pow), POWER_FACTOR)[0]
        
        return jsonify({"predicted_purchase_value": float(final_prediction)})
        
    except Exception as e:
        logger.error(f"Prediction Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    from waitress import serve
    logger.info("Starting Flask server on http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)
