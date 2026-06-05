import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
 
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
    "booking_changes", "days_in_waiting_list", "country"
]

 
# VALUE_BLOCK: economic value / reliability variables.
# Excluded from "no_value_block" representation and reserved for post-hoc
# profiling, following the professor's correction #2/#3.
VALUE_BLOCK = [
    "adr", "deposit_type",
    "previous_cancellations", "previous_bookings_not_canceled",
    "is_repeated_guest",
]

# PROFILE_BLOCK: booking-type categorization features.
# Describe channel/segment/product choice rather than booking complexity.
# Removed in "complexity_only" representation; reserved for post-hoc profiling.
PROFILE_BLOCK = [
    "market_segment", "distribution_channel", "customer_type",
    "reserved_room_type", "meal",
]

# Derived seasonal and property features, known at booking creation.
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
    "hotel_binary",
]

CYCLIC_FEATURES = [
    "arrival_month_sin", "arrival_month_cos"
]
 
_MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

WINSOR_CONFIG = {
    "lead_time": 0.99,
    "children":       0.99,
    "required_car_parking_spaces": 0.99,
    "adults":       0.99,
    "total_of_special_requests": 0.99,
    "babies":       0.99
}

# log(1+x) applied after winsorization to reduce right-skew before scaling.
LOG_TRANSFORM_COLS = [
    "lead_time",
    #"children",
    #"required_car_parking_spaces",
    #"adults",
    #"total_of_special_requests",
    #"babies"
]

 
RARE_THRESHOLD = 0.01



class MeanRangeScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=0)
        self.range_ = X.max(axis=0) - X.min(axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.range_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

 
 
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
 
# Replace categories whose frequency is below threshold with 'Other'
def _group_rare_categories(df: pd.DataFrame, cols: list, threshold: float) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].where(~df[col].isin(rare), other="Other")
    return df
 

def preprocess_data(df: pd.DataFrame, feature_set: str = "full", scaler: str = "standard"):
   
    df = df.copy()

    df = df.drop_duplicates()

    df = _derive_context_features(df)
    df = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df.columns])

    features = FULL_FEATURE_SET.copy()
    if feature_set == "no_value_block":
        features = [f for f in features if f not in VALUE_BLOCK]
    elif feature_set == "no_context":
        features = [f for f in features if f not in CONTEXT_BLOCK]
    elif feature_set == "complexity_only":
        features = [f for f in features if f not in VALUE_BLOCK and f not in PROFILE_BLOCK]
    elif feature_set != "full":
        raise ValueError(
            f"Unknown feature_set: {feature_set!r}. "
            "Expected 'full', 'no_value_block', 'no_context', or 'complexity_only'."
        )

    df = df[[f for f in features if f in df.columns]]

    if "children" in df.columns:
        df["children"] = df["children"].fillna(0.0)
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
 
    if scaler != "robust":  # robust scaler already handles outliers
        for col in WINSOR_CONFIG:
            if col in df.columns:
                bound = float(np.quantile(df[col].values, WINSOR_CONFIG[col]))
                df[col] = df[col].clip(upper=bound)

        for col in LOG_TRANSFORM_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col])

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df = _group_rare_categories(df, cat_cols, RARE_THRESHOLD)

    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")         # drop="if_binary" removes one redundant column for binary variables,
        ohe_array = ohe.fit_transform(df[cat_cols])
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        ohe_array = np.empty((len(df), 0))
        ohe_names = []

    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "robust":
        scaler_obj = RobustScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler == "mean":
        scaler_obj = MeanRangeScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler!r}. Expected 'standard' or 'robust'.")
    num_array = scaler_obj.fit_transform(df[num_cols].values.astype(np.float64))
 
    # Cyclic features are pre-normalised to [-1, 1] by sin/cos encoding; skip scaler.
    cyclic_cols = [c for c in CYCLIC_FEATURES if c in df.columns]
    cyclic_array = df[cyclic_cols].values.astype(np.float64) if cyclic_cols else np.empty((len(df), 0))

    X = np.hstack([num_array, ohe_array, cyclic_array])
    feature_names = num_cols + ohe_names + cyclic_cols

    return X, feature_names