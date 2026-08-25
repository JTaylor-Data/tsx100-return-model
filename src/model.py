"""LightGBM regressor (primary model) + Elastic Net and naive-momentum baselines.

The GBM is kept deliberately shallow/regularized: with ~100 stocks x a handful
of effectively-independent yearly snapshots (adjacent monthly rows are highly
autocorrelated), the true sample size is much smaller than the row count
suggests, so overfitting a flexible model is the main risk (Section 6).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "forward_return_12m"
NON_FEATURE_COLS = {"date", "yf_ticker", "ticker", "close", TARGET, "label_end_date"}


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def _lgb_dataset(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].copy()
    X["sector"] = X["sector"].astype("category")
    y = df[TARGET].values
    return X, y


def train_lightgbm(train_df: pd.DataFrame, feature_cols: list) -> lgb.LGBMRegressor:
    X, y = _lgb_dataset(train_df, feature_cols)
    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        num_leaves=15,
        min_child_samples=30,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X, y, categorical_feature=["sector"])
    return model


def predict_lightgbm(model: lgb.LGBMRegressor, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    X = df[feature_cols].copy()
    X["sector"] = X["sector"].astype("category")
    return model.predict(X)


def _elastic_net_pipeline(feature_cols: list) -> Pipeline:
    numeric_cols = [c for c in feature_cols if c != "sector"]
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
        ("sector", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000)),
    ])


def train_elastic_net(train_df: pd.DataFrame, feature_cols: list) -> Pipeline:
    pipe = _elastic_net_pipeline(feature_cols)
    pipe.fit(train_df[feature_cols], train_df[TARGET])
    return pipe


def predict_elastic_net(pipe: Pipeline, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    return pipe.predict(df[feature_cols])


def predict_naive_momentum(df: pd.DataFrame) -> np.ndarray:
    """Rank-by-trailing-12mo-momentum baseline -- not a return forecast, just a score for ranking."""
    return df["mom_12m"].values
