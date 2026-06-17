import numpy as np
import pandas as pd

# =============================================================================
# VALIDATION SPLITS
# =============================================================================

VALIDATION_SPLITS = [
    {
        "name": "Split_A",
        "train_start": "2011-01-29",
        "train_end": "2014-09-30",
        "valid_start": "2014-10-01",
        "valid_end": "2014-12-31",
    },
    {
        "name": "Split_B",
        "train_start": "2011-01-29",
        "train_end": "2015-06-30",
        "valid_start": "2015-07-01",
        "valid_end": "2015-09-30",
    }
]

# =============================================================================
# LEADERBOARD MIMIC
# =============================================================================

LEADERBOARD_PERIOD = {
    "train_start": "2011-01-29",
    "train_end": "2015-09-30",
    "forecast_start": "2015-10-01",
    "forecast_end": "2015-12-31",
}

# =============================================================================
# ROLLING BACKTEST WINDOWS
# =============================================================================

ROLLING_WINDOWS = [
    ("2014-01-01", "2014-03-31"),
    ("2014-04-01", "2014-06-30"),
    ("2014-07-01", "2014-09-30"),
    ("2014-10-01", "2014-12-31"),
    ("2015-01-01", "2015-03-31"),
    ("2015-04-01", "2015-06-30"),
    ("2015-07-01", "2015-09-30"),
]

# =============================================================================
# METRICS
# =============================================================================

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true > 0 # exclude closed/near-closed days

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


def wape(y_true, y_pred):
    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(y_true - y_pred))
        / denominator
        * 100
    )


# =============================================================================
# GLOBAL METRICS
# =============================================================================

def compute_metrics(y_true, y_pred):

    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
    }


# =============================================================================
# PER STORE METRICS
# =============================================================================

def compute_store_metrics(
    valid_df,
    predictions,
    target_col="revenue",
):

    temp = valid_df.copy()
    temp["prediction"] = predictions

    rows = []

    for store_id, grp in temp.groupby("store_id"):

        rows.append(
            {
                "store_id": store_id,
                "MAE": mae(
                    grp[target_col],
                    grp["prediction"]
                ),
                "RMSE": rmse(
                    grp[target_col],
                    grp["prediction"]
                ),
                "MAPE": mape(
                    grp[target_col],
                    grp["prediction"]
                ),
                "WAPE": wape(
                    grp[target_col],
                    grp["prediction"]
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# SPLIT CREATION
# =============================================================================

def create_split(
    df,
    train_start,
    train_end,
    valid_start,
    valid_end,
):

    train_df = df[
        (df["date"] >= train_start)
        & (df["date"] <= train_end)
    ].copy()

    valid_df = df[
        (df["date"] >= valid_start)
        & (df["date"] <= valid_end)
    ].copy()

    return train_df, valid_df


# =============================================================================
# STANDARD VALIDATION
# =============================================================================

def run_validation_split(
    df,
    split_config,
):

    train_df, valid_df = create_split(
        df,
        split_config["train_start"],
        split_config["train_end"],
        split_config["valid_start"],
        split_config["valid_end"],
    )

    return train_df, valid_df


# =============================================================================
# ROLLING ORIGIN SPLITS
# =============================================================================

def rolling_backtest_generator(
    df,
    windows=ROLLING_WINDOWS,
):

    for valid_start, valid_end in windows:

        train_df = df[
            df["date"] < valid_start
        ].copy()

        valid_df = df[
            (df["date"] >= valid_start)
            & (df["date"] <= valid_end)
        ].copy()

        yield (
            valid_start,
            valid_end,
            train_df,
            valid_df,
        )


# =============================================================================
# RESULT AGGREGATION
# =============================================================================

def summarize_results(results):

    return (
        pd.DataFrame(results)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )


# =============================================================================
# LEADERBOARD SIMULATION
# =============================================================================

def get_leaderboard_training_data(df):

    train_df = df[
        (df["date"] >= LEADERBOARD_PERIOD["train_start"])
        & (df["date"] <= LEADERBOARD_PERIOD["train_end"])
    ].copy()

    return train_df


def get_leaderboard_forecast_period():

    return (
        LEADERBOARD_PERIOD["forecast_start"],
        LEADERBOARD_PERIOD["forecast_end"],
    )