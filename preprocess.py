import joblib
import numpy as np
import pandas as pd

SEASON_MAP = {
    12: 1, 1: 1, 2: 1,   # Winter
    3: 2, 4: 2, 5: 2,    # Spring
    6: 3, 7: 3, 8: 3,    # Summer
    9: 4, 10: 4, 11: 4   # Autumn
}


def _validate_min_days(raw_df: pd.DataFrame, min_days: int = 90) -> None:
    """Require at least `min_days` unique dates per station."""
    tmp = raw_df.copy()
    tmp["Datetime"] = pd.to_datetime(tmp["Datetime"], errors="coerce")
    tmp = tmp.dropna(subset=["Datetime"])

    day_count = (
        tmp.assign(_date=tmp["Datetime"].dt.normalize())
           .groupby("Hydro ID")["_date"]
           .nunique()
    )

    too_short = day_count[day_count < min_days]
    if not too_short.empty:
        detail = ", ".join([f"{k}:{int(v)}d" for k, v in too_short.items()])
        raise ValueError(
            f"New data must contain at least {min_days} days per station. "
            f"Too short -> {detail}"
        )


def preprocess_raw(raw_df: pd.DataFrame, bundle: dict, min_days: int = 90) -> pd.DataFrame:
    """
    Input: raw data (must include Datetime, Hydro ID, precip)
    Output: feature matrix aligned with training columns
    """
    required = {"Datetime", "Hydro ID", "precip"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Enforce minimum data length per station
    _validate_min_days(raw_df, min_days=min_days)

    df = raw_df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values(["Hydro ID", "Datetime"])

    # Rolling precipitation features: shift(1) then rolling sum (same as training)
    for w in [3, 7, 14, 30]:
        df[f"precip_sum_{w}d"] = (
            df.groupby("Hydro ID")["precip"]
              .transform(lambda s: s.shift(1).rolling(w, min_periods=1).sum())
        )

    roll_cols = ["precip_sum_3d", "precip_sum_7d", "precip_sum_14d", "precip_sum_30d"]
    df[roll_cols] = df[roll_cols].fillna(0)

    # Time features
    df["month"] = df["Datetime"].dt.month
    df["dayofyear"] = df["Datetime"].dt.dayofyear
    df["season_code"] = df["month"].map(SEASON_MAP)

    year_days = np.where(df["Datetime"].dt.is_leap_year, 366, 365)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / year_days)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / year_days)

    # Rain feature transform: log1p + MinMaxScaler from bundle
    rain_cols = ["precip", "precip_sum_3d", "precip_sum_7d", "precip_sum_14d", "precip_sum_30d"]
    df[rain_cols] = np.log1p(df[rain_cols].clip(lower=0))
    mm = bundle["mm_scaler"]
    df[rain_cols] = mm.transform(df[rain_cols])

    # One-hot encoding (same pattern as training)
    df = pd.get_dummies(
        df,
        columns=["Hydro ID", "season_code"],
        prefix=["HydroID", "season"],
        dtype=int
    )

    # Drop raw time columns not used by the model
    df = df.drop(columns=[c for c in ["Datetime", "month", "dayofyear"] if c in df.columns], errors="ignore")

    # Align to training feature columns: add missing as 0, drop extras, lock order
    feature_columns = bundle["feature_columns"]
    X = df.reindex(columns=feature_columns, fill_value=0).astype(np.float32)
    return X


def predict_from_raw(raw_df: pd.DataFrame, pkl_path: str, min_days: int = 90) -> np.ndarray:
    bundle = joblib.load(pkl_path)
    model = bundle["model"]
    X = preprocess_raw(raw_df, bundle, min_days=min_days)
    return model.predict(X)


def predict_with_station_qflag(raw_df: pd.DataFrame, pkl_path: str) -> pd.DataFrame:
    """
    Use fixed station-wise 0.75 quantile for high-risk flag:
    pred >= pred_q75 => high risk
    """
    bundle = joblib.load(pkl_path)
    model = bundle["model"]

    # Require at least 90 days per station in new data
    _validate_min_days(raw_df, min_days=90)

    df = raw_df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values(["Hydro ID", "Datetime"]).reset_index(drop=True)

    X = preprocess_raw(df, bundle, min_days=90)
    pred = model.predict(X)

    out = df[["Datetime", "Hydro ID"]].copy()
    out["pred"] = pred

    # Fixed 0.75 threshold per station, computed within the new-data prediction distribution
    out["pred_q75"] = out.groupby("Hydro ID")["pred"].transform(lambda s: s.quantile(0.75))

    # If prediction reaches/exceeds station q75, risk is above normal period
    out["is_high_risk_q75"] = out["pred"] >= out["pred_q75"]

    return out


if __name__ == "__main__":
    # Example usage
    df_new = pd.read_csv("new_data.csv")  # Required columns: Datetime, Hydro ID, precip
    out = predict_with_station_qflag(df_new, "artifacts/cb_tplus1d_7d_only.pkl")
    print(out.head(10).to_string(index=False))
    out.to_csv("pred_with_station_q75_flag.csv", index=False)