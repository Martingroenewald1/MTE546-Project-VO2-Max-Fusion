import pandas as pd
import numpy as np

XLSX_PATH = "sweat_lactate_data/Sweat_lactate_data.xlsx"


# load info
def load_participants(xlsx_path):
    df = pd.read_excel(xlsx_path, sheet_name="Participant", engine="openpyxl")

    # clean column names (remove weird spacing)
    df.columns = df.columns.str.strip()

    # rename important columns
    df = df.rename(columns={
        "Participant number": "participant_id",
        "Weighe": "weight",  # typo in dataset
        "VO2max": "vo2max",
        "Group": "group",
        "Age": "age"
    })

    # keep only relevant columns
    df = df[["participant_id", "group", "age", "weight", "vo2max"]].copy()

    # keep only real participants (remove avg/sd rows)
    df = df[df["group"].isin(["Untrained", "Trained"])].copy()

    return df


# sweat table timeseries
def build_sweat_timeseries(xlsx_path, participants):

    # load messy wide-format sheet
    wide_df = pd.read_excel(
        xlsx_path,
        sheet_name="Sweat rate and Sweat lactate",
        header=None,
        engine="openpyxl"
    )

    # time column (starts at row 4)
    time_vals = pd.to_numeric(wide_df.iloc[4:, 0], errors="coerce").to_numpy()

    records = []

    # split participants by group
    untrained_ids = participants.loc[
        participants["group"] == "Untrained", "participant_id"
    ].tolist()

    trained_ids = participants.loc[
        participants["group"] == "Trained", "participant_id"
    ].tolist()

    # theres a weird block between trained and untrained we have to ignore
    # columns: 1,2 then 3,4 for 15 people
    for i, pid in enumerate(untrained_ids):
        sr = pd.to_numeric(wide_df.iloc[4:, 1 + 2*i], errors="coerce").to_numpy()
        la = pd.to_numeric(wide_df.iloc[4:, 2 + 2*i], errors="coerce").to_numpy()

        temp = pd.DataFrame({
            "participant_id": pid,
            "group": "Untrained",
            "time": time_vals,
            "sweat_rate": sr,
            "lactate": la         
        })

        records.append(temp)

    # TRAINED BLOCK starts at columns 38,39
    for i, pid in enumerate(trained_ids):
        sr = pd.to_numeric(wide_df.iloc[4:, 38 + 2*i], errors="coerce").to_numpy()
        la = pd.to_numeric(wide_df.iloc[4:, 39 + 2*i], errors="coerce").to_numpy()

        temp = pd.DataFrame({
            "participant_id": pid,
            "group": "Trained",
            "time": time_vals,
            "sweat_rate": sr,
            "lactate": la
        })

        records.append(temp)

    # combine all participants
    sweat_ts = pd.concat(records, ignore_index=True)

    # remove invalid rows
    sweat_ts = sweat_ts.dropna(subset=["time"]).copy()

    return sweat_ts


# 3) load threshold and features
def load_threshold_features(xlsx_path, participants):

    df = pd.read_excel(xlsx_path, sheet_name="Threshold and slope", engine="openpyxl")

    # keep only participant rows
    df = df.iloc[1:31].copy().reset_index(drop=True)

    # assign group manually (first 15 untrained, next 15 trained)
    df["group"] = ["Untrained"] * 15 + ["Trained"] * 15

    # align participant IDs with main sheet
    df["participant_id"] = participants["participant_id"].tolist()

    # unify threshold column (dataset split it weirdly)
    df["sweat_rate_threshold"] = df["Sweat rate threshold\n(mg/cm2/min)"]

    df.loc[df["group"] == "Untrained", "sweat_rate_threshold"] = df.loc[
        df["group"] == "Untrained", "Unnamed: 5"
    ]

    # rename useful columns
    df = df.rename(columns={
        "Preferred model": "preferred_model",
        "Breakpoint type": "breakpoint_type",
        "Slope for simple linear regression\nmg/cm2/min per unit sweat rate increase": "sheet_slope"
    })

    return df[[
        "participant_id",
        "group",
        "preferred_model",
        "breakpoint_type",
        "sweat_rate_threshold",
        "sheet_slope"
    ]].copy()


# slope function (time trend)
def slope_or_nan(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # remove NaNs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    return np.polyfit(x, y, 1)[0]


# sweat features csv
def build_sweat_features(sweat_ts):

    results = []

    for pid in sweat_ts["participant_id"].unique():

        # isolate one participant
        data = sweat_ts[sweat_ts["participant_id"] == pid].copy()
        data = data.sort_values("time")

        lactate = data["lactate"].to_numpy()
        sweat_rate = data["sweat_rate"].to_numpy()
        time = data["time"].to_numpy()

        results.append({
            "participant_id": pid,
            "group": data["group"].iloc[0],

            # lactate features
            "lactate_mean": np.nanmean(lactate),
            "lactate_max": np.nanmax(lactate),
            "lactate_std": np.nanstd(lactate),
            "lactate_final": lactate[-1],
            "lactate_slope_time": slope_or_nan(time, lactate),

            # sweat rate features
            "sweat_rate_mean": np.nanmean(sweat_rate),
            "sweat_rate_max": np.nanmax(sweat_rate),
            "sweat_rate_std": np.nanstd(sweat_rate),
            "sweat_rate_final": sweat_rate[-1],
            "sweat_rate_slope_time": slope_or_nan(time, sweat_rate),
        })

    return pd.DataFrame(results)


# it all comes together i promise
participants = load_participants(XLSX_PATH)

sweat_ts = build_sweat_timeseries(XLSX_PATH, participants)

threshold_features = load_threshold_features(XLSX_PATH, participants)

sweat_features = build_sweat_features(sweat_ts)

# merge everything together
final_sweat_features = (
    participants
    .merge(sweat_features, on=["participant_id", "group"], how="inner")
    .merge(threshold_features, on=["participant_id", "group"], how="left")
)

# save outputs
sweat_ts.to_csv("sweat_timeseries.csv", index=False)
final_sweat_features.to_csv("sweat_features.csv", index=False)