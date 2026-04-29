import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

# Dropped before any feature selection — post-event or high-cardinality identifiers.
# booking_changes and days_in_waiting_list are excluded because they accumulate
# after the booking is created (leakage under booking-creation segmentation time).
EXCLUDED_COLUMNS = [
    "is_canceled", "reservation_status", "reservation_status_date",
    "assigned_room_type",
    "agent", "company",
    "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month",
    "booking_changes", "days_in_waiting_list",
]

# TODO: Add POSTHOC_COLUMNS variable?

VALUE_BLOCK = [
    "adr", "deposit_type",
    "previous_cancellations", "previous_bookings_not_canceled",
    "is_repeated_guest",
]

# Derived from hotel and arrival_date_month, known at booking creation, so we do not remove temporal/property context a priori.
CONTEXT_BLOCK = ["arrival_month_sin", "arrival_month_cos", "hotel_binary"]

FULL_FEATURE_SET = [
    "lead_time",
    "adults", "children", "babies",
    "market_segment", "distribution_channel", "customer_type", "adr",
    "previous_cancellations", "previous_bookings_not_canceled",
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

RARE_THRESHOLD = 0.01


def _derive_context_features(df):
    df = df.copy()
    if "hotel" in df.columns:
        df["hotel_binary"] = (df["hotel"] == "Resort Hotel").astype(float)
    if "arrival_date_month" in df.columns:
        month_num = df["arrival_date_month"].map(_MONTH_MAP)
        df["arrival_month_sin"] = np.sin(2 * np.pi * month_num / 12)
        df["arrival_month_cos"] = np.cos(2 * np.pi * month_num / 12)
    return df


def _group_rare_categories(df, cols, threshold):
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].where(~df[col].isin(rare), other="Other")
    return df


def preprocess_data(df, feature_set="full", scaler="standard"):
    df = df.copy()

    df = _derive_context_features(df)

    df = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df.columns])

    # Feature set
    features = FULL_FEATURE_SET.copy()
    if feature_set == "no_value_block":
        features = [f for f in features if f not in VALUE_BLOCK]
    elif feature_set == "no_context":
        features = [f for f in features if f not in CONTEXT_BLOCK]
    elif feature_set != "full":
        raise ValueError(f"Unknown feature_set: {feature_set!r}. "
                         "Expected 'full', 'no_value_block', or 'no_context'.")
    df = df[[f for f in features if f in df.columns]]

    # Imputation
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0.0)
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Group rare categories before OHE
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df = _group_rare_categories(df, cat_cols, RARE_THRESHOLD)

    # Categorical encoding (we only need OHE)
    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False)
        ohe_array = ohe.fit_transform(df[cat_cols])
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        ohe_array = np.empty((len(df), 0))
        ohe_names = []

    # Numerical scaling (TODO: Add more scalers to experiment with)
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "robust":
        scaler_obj = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler!r}. Expected 'standard' or 'robust'.")
    num_array = scaler_obj.fit_transform(df[num_cols].values.astype(np.float64))
    
    X = np.hstack([num_array, ohe_array])
    feature_names = num_cols + ohe_names
    return X, feature_names
