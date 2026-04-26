import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

EXCLUDED_COLUMNS = [
    "is_canceled", "reservation_status", "reservation_status_date",
    "agent", "company",
    "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month",
]

VALUE_BLOCK = [
    "adr", "deposit_type",
    "previous_cancellations", "previous_bookings_not_canceled",
    "is_repeated_guest",
]

FULL_FEATURE_SET = [
    "lead_time", "booking_changes", "days_in_waiting_list",
    "adults", "children", "babies",
    "market_segment", "distribution_channel", "customer_type", "adr",
    "previous_cancellations", "previous_bookings_not_canceled",
    "deposit_type", "is_repeated_guest",
    "total_of_special_requests", "meal", "reserved_room_type",
    "required_car_parking_spaces",
]

CATEGORICAL_FEATURES = [
    "market_segment", "distribution_channel", "customer_type",
    "deposit_type", "meal", "reserved_room_type",
]

NUMERICAL_FEATURES = [
    "lead_time", "booking_changes", "days_in_waiting_list",
    "adults", "children", "babies",
    "previous_cancellations", "previous_bookings_not_canceled",
    "adr", "total_of_special_requests",
    "required_car_parking_spaces", "is_repeated_guest",
]

RARE_THRESHOLD = 0.01


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

    # 1. Drop excluded columns
    df = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df.columns])

    # 2. Select feature set
    # TODO: We could probably specify the columns in the args directly
    features = FULL_FEATURE_SET.copy()
    if feature_set == "no_value_block":
        features = [f for f in features if f not in VALUE_BLOCK]
    df = df[[f for f in features if f in df.columns]]

    # 3. Impute missing values
    # I assume missing values in the children column are 0, but maybe we should compare imputation strategies
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0.0)
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 4. Group rare categories before OHE
    # TODO: This was a suggestion of which I am not sure about yet
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df = _group_rare_categories(df, cat_cols, RARE_THRESHOLD)

    # 5. One-hot encode categoricals
    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe_array = ohe.fit_transform(df[cat_cols])
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        ohe_array = np.empty((len(df), 0))
        ohe_names = []

    # 6. Scale numericals
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "robust":
        scaler_obj = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler!r}. Expected 'standard' or 'robust'.")
    num_array = scaler_obj.fit_transform(df[num_cols].values)

    # 7. Concatenate
    X = np.hstack([num_array, ohe_array])
    feature_names = num_cols + ohe_names
    return X, feature_names
