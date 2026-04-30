import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
 
#TODO TALVEZ METER A JUSTIFICAÇÃO APENAS NO RELATORIO E ABREVIAR OS COMENTARIS 
# Columns excluded before any feature selection.
# Leakage: post-event outcomes (is_canceled, reservation_status*).
# Segmentation-time: booking_changes and days_in_waiting_list accumulate
#   after booking creation, so excluded under booking-creation index time.
# High cardinality: agent, company (too many IDs, no behavioural signal).
# Temporal granularity: week number and day of month excluded; month is
#   retained via cyclic encoding to preserve coarse seasonality (correction #4).
EXCLUDED_COLUMNS = [
    "is_canceled", "reservation_status", "reservation_status_date",
    "assigned_room_type",
    "agent", "company",
    "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month",
    "booking_changes", "days_in_waiting_list",
]
 
# VALUE_BLOCK: economic value / reliability variables.
# Excluded from "no_value_block" representation and reserved for post-hoc
# profiling, following the professor's correction #2/#3.
VALUE_BLOCK = [
    "adr", "deposit_type",
    "previous_cancellations", "previous_bookings_not_canceled",
    "is_repeated_guest",
]
 
# Derived seasonal and property features, known at booking creation.
# Included to avoid suppressing genuine seasonality effects (correction #4).
CONTEXT_BLOCK = ["arrival_month_sin", "arrival_month_cos", "hotel_binary"]
 
FULL_FEATURE_SET = [
    "lead_time",
    "adults", "children", "babies",
    "market_segment", "distribution_channel", "customer_type",
    "adr", "previous_cancellations", "previous_bookings_not_canceled",
    "deposit_type", "is_repeated_guest",
    "total_of_special_requests", "meal", "reserved_room_type",
    "required_car_parking_spaces",
    "arrival_month_sin", "arrival_month_cos", "hotel_binary",
]
 
CATEGORICAL_FEATURES = [
    "market_segment", "distribution_channel", "customer_type",
    "deposit_type", "meal", "reserved_room_type",
]
 
NUMERICAL_FEATURES = [
    "lead_time",
    "adults", "children", "babies",
    "previous_cancellations", "previous_bookings_not_canceled",
    "adr", "total_of_special_requests",
    "required_car_parking_spaces", "is_repeated_guest",
    "arrival_month_sin", "arrival_month_cos", "hotel_binary",
]
 
_MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

WINSOR_CONFIG = {
    "lead_time": 0.99,
    "adr":       0.99,
}

 
RARE_THRESHOLD = 0.01
 
 
def _derive_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic month encoding (sin/cos) and binary hotel indicator.
    Cyclic encoding avoids a discontinuity between December and January."""
    df = df.copy()
    if "hotel" in df.columns:
        df["hotel_binary"] = (df["hotel"] == "Resort Hotel").astype(float)
    if "arrival_date_month" in df.columns:
        month_num = df["arrival_date_month"].map(_MONTH_MAP)
        df["arrival_month_sin"] = np.sin(2 * np.pi * month_num / 12)
        df["arrival_month_cos"] = np.cos(2 * np.pi * month_num / 12)
    return df
 
 
def _group_rare_categories(df: pd.DataFrame, cols: list, threshold: float) -> pd.DataFrame:
    """Replace categories whose frequency is below threshold with 'Other'."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].where(~df[col].isin(rare), other="Other")
    return df
 
# Fit and transform a training DataFrame into a numeric feature matrix to feed the models ( → impute missing values → group rare categories → OHE → scale → return X.)
# Parameters:
# feature_set : "full" | "no_value_block" | "no_context"
# scaler      : "standard" (Lab4 default) | "robust"
def preprocess_data(df: pd.DataFrame, feature_set: str = "full", scaler: str = "standard"):
   
    df = df.copy()
 
    df = _derive_context_features(df)
    df = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df.columns])
 
    features = FULL_FEATURE_SET.copy()
    if feature_set == "no_value_block":
        features = [f for f in features if f not in VALUE_BLOCK]
    elif feature_set == "no_context":
        features = [f for f in features if f not in CONTEXT_BLOCK]
    elif feature_set != "full":
        raise ValueError(
            f"Unknown feature_set: {feature_set!r}. "
            "Expected 'full', 'no_value_block', or 'no_context'."
        )
    df = df[[f for f in features if f in df.columns]]
 
 
    dup_mask = df.T.duplicated()
    dup_cols = df.columns[dup_mask].tolist()
    if dup_cols:
        print(f"[preprocess_data] Dropped {len(dup_cols)} exact duplicate column(s): {dup_cols}")
        df = df.loc[:, ~dup_mask]
 
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0.0)
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
 
    winsor_bounds = {}
    for col in WINSOR_CONFIG:
        if col in df.columns:
            bound = float(np.quantile(df[col].values, WINSOR_CONFIG[col]))
            df[col] = df[col].clip(upper=bound)
            winsor_bounds[col] = bound
 
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df = _group_rare_categories(df, cat_cols, RARE_THRESHOLD)
 
    if cat_cols:
        # drop="if_binary" removes one redundant column for binary variables,
        # avoiding double-counting in Euclidean distance.
        ohe       = OneHotEncoder(sparse_output=False, drop="if_binary")
        ohe_array = ohe.fit_transform(df[cat_cols])
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        ohe       = None
        ohe_array = np.empty((len(df), 0))
        ohe_names = []
 
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "robust":
        scaler_obj = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler!r}. Expected 'standard' or 'robust'.")
    num_array = scaler_obj.fit_transform(df[num_cols].values.astype(np.float64))
 
    X             = np.hstack([num_array, ohe_array])
    feature_names = num_cols + ohe_names
    
    return X, feature_names