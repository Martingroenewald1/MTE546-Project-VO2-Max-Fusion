import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import HuberRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from sklearn.linear_model import Ridge, Lasso



def evaluate_sweat_only_vo2max(feature_csv_path):
    
    # get csv
    df = pd.read_csv(feature_csv_path)
    df.columns = df.columns.str.strip()
    df["sweat_rate_threshold"] = pd.to_numeric(
        df["sweat_rate_threshold"].replace("-", np.nan),
        errors="coerce"
    )
    # define inputs and target
    target_col = "vo2max"

    feature_cols = [
        "age",
        "weight",
        "lactate_mean",
        "lactate_max",
    ]

    # ensure numeric
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # drop rows with missing inputs OR target
    df = df.dropna(subset=feature_cols + [target_col])
    results = []

    #diff models that can be used
    models = {
        "ridge": Ridge(),
        "lasso": Lasso(max_iter=5000),
        "huber": HuberRegressor(max_iter=1000),
    }

    # different combos that may give better results

    feature_sets = {
        "lactate_basic": [
            "lactate_mean",
            "lactate_max",
        ],

        "lactate_plus_body": [
            "age",
            "weight",
            "lactate_mean",
            "lactate_max",
        ],

        "sweat_basic": [
            "sweat_rate_mean",
            "sweat_rate_max",
        ],

        "sweat_trend": [
            "sweat_rate_mean",
            "sweat_rate_std",
            "sweat_rate_slope_time",
        ],

        "sweat_threshold": [
            "sweat_rate_max",
            "sweat_rate_threshold",
        ],

        "mixed_small_1": [
            "lactate_mean",
            "sweat_rate_max",
        ],

        "mixed_small_2": [
            "lactate_max",
            "sweat_rate_mean",
            "sweat_rate_max",
        ],

        "mixed_small_3": [
            "lactate_mean",
            "sweat_rate_std",
        ],
    }

    all_results = []


    for feat_name, feature_cols in feature_sets.items():
        for model_name, reg in models.items():

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", reg)
            ])

            # clean per feature set
            df_sub = df.copy()
            df_sub[feature_cols] = df_sub[feature_cols].apply(pd.to_numeric, errors="coerce")
            df_sub[target_col] = pd.to_numeric(df_sub[target_col], errors="coerce")
            df_sub = df_sub.dropna(subset=feature_cols + [target_col])

            abs_errors = []
            y_true_all = []
            y_pred_all = []

            for pid in df_sub["participant_id"].unique():
                train_df = df_sub[df_sub["participant_id"] != pid]
                test_df = df_sub[df_sub["participant_id"] == pid]

                X_train = train_df[feature_cols]
                y_train = train_df[target_col]

                X_test = test_df[feature_cols]
                y_true = float(test_df[target_col].iloc[0])

                model.fit(X_train, y_train)
                y_pred = float(model.predict(X_test)[0])

                abs_errors.append(abs(y_pred - y_true))
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)

            mae = np.mean(abs_errors)
            rmse = np.sqrt(np.mean((np.array(y_pred_all) - np.array(y_true_all))**2))
            r2 = r2_score(y_true_all, y_pred_all)

            all_results.append({
                "features": feat_name,
                "model": model_name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            })

    results_df = pd.DataFrame(all_results).sort_values("mae")
    print(results_df)

    return results_df

results_df = evaluate_sweat_only_vo2max("sweat_features.csv")
results_df.to_csv("sweat_only_results.csv", index=False)

