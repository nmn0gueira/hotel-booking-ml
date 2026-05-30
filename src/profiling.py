import numpy as np
import pandas as pd

POSTHOC_COLS = [
    "adr", "deposit_type", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "is_canceled", "booking_changes",
]


def compute_cluster_profiles(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with rows = clusters, columns = per-feature mean/std + size.
    If raw_df is provided, post-hoc variables (adr, is_canceled, etc.) are appended
    using the original unscaled values — these are profiling-only, not clustering inputs.
    """
    df_X = pd.DataFrame(X, columns=feature_names)
    df_X["_cluster"] = labels

    rows = []
    for cluster_id in sorted(df_X["_cluster"].unique()):
        mask = df_X["_cluster"] == cluster_id
        subset = df_X[mask].drop(columns=["_cluster"])
        row = {"cluster": int(cluster_id), "size": int(mask.sum())}
        for col in feature_names:
            vals = subset[col].values
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals))
        rows.append(row)

    profile = pd.DataFrame(rows).set_index("cluster")

    if raw_df is not None:
        posthoc_present = [c for c in POSTHOC_COLS if c in raw_df.columns]
        df_raw = raw_df.copy().iloc[: len(labels)].reset_index(drop=True)
        df_raw["_cluster"] = labels
        for cluster_id in sorted(profile.index):
            subset_raw = df_raw[df_raw["_cluster"] == cluster_id]
            for col in posthoc_present:
                if pd.api.types.is_numeric_dtype(subset_raw[col]):
                    profile.loc[cluster_id, f"posthoc_{col}_mean"] = float(
                        subset_raw[col].mean()
                    )
                else:
                    mode_val = subset_raw[col].mode()
                    profile.loc[cluster_id, f"posthoc_{col}_mode"] = (
                        mode_val.iloc[0] if len(mode_val) > 0 else ""
                    )

    return profile
