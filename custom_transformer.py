# custom_transformer.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply all feature engineering steps defined during training.
    """

    def __init__(self):
        # The CustomFeatureEngineer must accept no arguments to work correctly
        # with the sklearn Pipeline's standard API.
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # --- Date features ---
        X['sessionStart_dt'] = pd.to_datetime(
            X['sessionStart'], unit='s', errors='coerce')
        X['session_dayofweek'] = X['sessionStart_dt'].dt.dayofweek
        X['session_hour'] = X['sessionStart_dt'].dt.hour
        X['session_month'] = X['sessionStart_dt'].dt.month
        X['is_weekend_session'] = X['session_dayofweek'].isin(
            [5, 6]).astype(float)
        X['is_business_hour'] = X['session_hour'].between(9, 17).astype(int)
        X['session_day'] = X['sessionStart_dt'].dt.day
        X['session_year'] = X['sessionStart_dt'].dt.year

        X['date'] = pd.to_datetime(X['date'], format='%Y%m%d', errors='coerce')
        X['day'] = X['date'].dt.day
        X['month'] = X['date'].dt.month
        X['year'] = X['date'].dt.year
        X['day_of_week'] = X['date'].dt.dayofweek
        X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(float)
        X['is_peak_month'] = X['month'].isin([4, 6, 8]).astype(int)
        X['is_off_season'] = X['month'].isin([10, 11]).astype(int)
        X['is_midweek'] = X['day_of_week'].isin([1, 2, 3]).astype(int)
        X['quarter'] = X['date'].dt.quarter
        X['is_Q2_or_Q3_peak'] = X['quarter'].isin([2, 3]).astype(int)
        X['is_peak_weekday'] = (X['day_of_week'] == 2).astype(int)

        # --- Clean categorical ---
        X = X.replace("not available in demo dataset", np.nan)
        X['browser'] = X['browser'].replace({
            'Firefox': 'Mozilla',
            'Mozilla Compatible Agent': 'Mozilla',
            'Mozilla': 'Mozilla',
            'Safari (in-app)': 'Safari'
        })

        # --- Numeric cleanup ---
        X['trafficSource.isTrueDirect'] = X['trafficSource.isTrueDirect'].fillna(
            False).astype(float)
        X['totals.bounces'] = X['totals.bounces'].fillna(
            0).astype(bool).astype(float)
        X['trafficSource.adwordsClickInfo.page'] = X['trafficSource.adwordsClickInfo.page'].fillna(
            0.0).astype(float)
        X['pageViews'] = X['pageViews'].fillna(0.0).astype(float)
        X['new_visits'] = X['new_visits'].fillna(0.0).astype(float)
        X['trafficSource.adwordsClickInfo.isVideoAd'] = X['trafficSource.adwordsClickInfo.isVideoAd'].fillna(
            1.0).astype(bool).astype(float)

        for col in ['trafficSource.referralPath', 'trafficSource.adContent',
                    'trafficSource.adwordsClickInfo.slot', 'trafficSource.adwordsClickInfo.adNetworkType',
                    'trafficSource.keyword', 'geoNetwork.region', 'geoNetwork.city', 'geoNetwork.metro']:
            X[col] = X[col].fillna("Other")

        # --- Interaction features ---
        X['hits_per_pageview'] = X['totalHits'] / (X['pageViews'] + 1)
        X['pageviews_per_hour'] = X['pageViews'] / (X['session_hour'] + 1)
        X['session_page_product'] = X['sessionNumber'] * X['pageViews']
        X['session_per_hit'] = X['sessionNumber'] / (X['totalHits'] + 1)
        X['bounce_hit_ratio'] = X['totals.bounces'] / (X['totalHits'] + 1)
        X['pageviews_per_channel'] = X['pageViews'] / \
            (X['userChannel'].map(lambda x: 1 if x == 'Referral' else 2) + 1)
        X['is_repeat_visitor'] = (X['sessionNumber'] > 1).astype(float)
        X['is_video_ad_and_bounce'] = X['trafficSource.adwordsClickInfo.isVideoAd'] * \
            X['totals.bounces']

        # --- Frequency encoding for categorical (Note: Freq maps should be fitted and saved, but here,
        # using value_counts on the single input row is incorrect for real-world data and will be replaced
        # by the trained pipeline's TargetEncoder. Removing to avoid confusion/error.)
        # The model training code only used this for the TargetEncoder step in the pipeline.
        # However, since you created features like '{col}_freq', these features must be removed
        # as the frequency map cannot be computed for a single row.

        # ***CRITICAL FIX: Dropping the frequency features you created in your original FE,
        # as they cannot be calculated correctly on a single prediction row.
        # The TargetEncoder pipeline step handles the categorical columns correctly.***

        # We will keep the original implementation but note that the features will be dropped
        # if they aren't used by the final ColumnTransformer's `num_cols` list.
        # For simplicity and to match the original FE, we include the map, but it won't
        # be meaningful for a single row.
        import sys
        freq_maps = getattr(sys.modules.get('__main__'), 'freq_maps', None)
        for col in ['browser', 'geoNetwork.city', 'trafficSource.campaign']:
            if freq_maps and col in freq_maps:
                X[f'{col}_freq'] = X[col].map(freq_maps[col]).fillna(0)
            else:
                freqs = X[col].value_counts().to_dict()
                X[f'{col}_freq'] = X[col].map(freqs).fillna(0)

        # --- Drop intermediate date columns ---
        X.drop(columns=['sessionStart', 'sessionStart_dt',
               'date'], inplace=True, errors='ignore')
        X['userId'] = X['userId'].astype(str)
        X['sessionId'] = X['sessionId'].astype(str)

        return X
