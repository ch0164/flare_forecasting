import pandas as pd

from source.common_imports import *
from source.constants import *


def get_median_df(cols):
    df = pd.DataFrame()
    for col in cols:
        temp_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_mx_data_{col.lower()}.csv")
        df[col] = temp_df["T_REC"]
    df = df.T
    date_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_dataset.csv")
    date_df = date_df.loc[date_df["AR_class"] == 1]
    dates = []
    for i in range(df.shape[1]):
        dates.append(df[i].mode(dropna=True)[0])
    date_df.insert(loc=0, column="T_REC", value=dates)
    print(date_df)
    date_df.to_csv(f"{FLARE_DATA_DIRECTORY}sinha_mx_data_majority.csv", index=False)
    return date_df


def get_flare_info(df):
    info_df = pd.read_csv(f"{FLARE_LIST_DIRECTORY}mx_list.txt")
    info_df["time_peak"] = pd.to_datetime(info_df["time_peak"])
    df["T_REC"] = df["T_REC"].str.replace(".000_TAI", "")
    df["T_REC"] = df["T_REC"].str.replace(".", "-")
    df["T_REC"] = df["T_REC"].str.replace("_", " ")
    df["T_REC"] = pd.to_datetime(df["T_REC"])
    df["time_peak"] = df["T_REC"] + pd.Timedelta(days=1)
    df.sort_values(by="time_peak", inplace=True)
    info_df.sort_values(by="time_peak", inplace=True)
    recorded_times_df = pd.merge_asof(df, info_df, by="time_peak", direction="nearest", tolerance=pd.Timedelta(hours=24))
    recorded_times_df.to_csv(f"{FLARE_LIST_DIRECTORY}sinha_flare_times.csv", index=False)


def main():
    sinha_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_dataset.csv")
    flare_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}mx_data.txt", header=0, delimiter=r"\s+")

    cols = ["TOTUSJH", "TOTUSJZ", "SAVNCPP", "R_VALUE", "SHRGT45", "ABSNJZH", "TOTPOT", "AREA_ACR", "USFLUX"]
    for col in cols:
        sinha_params = [
            "MEANPOT_y",
            "EPSZ",
            "TOTBSQ",
            "TOTFZ",
            "TOTABSTWIST"
        ]
        renamed_params = [
            "MEANPOT",
            "EPSZ",
            "TOTBSQ",
            "TOTFZ",
            "TOTABSTWIST"
        ]
        all_params = ["index", "T_REC", "NOAA_AR", col] + sinha_params

        sinha_df1 = sinha_df.loc[sinha_df["AR_class"] == 1]
        sinha_df1.reset_index(inplace=True)

        sinha_df1.sort_values(by=col, inplace=True)
        flare_df1 = flare_df.sort_values(by=col)

        df = pd.merge_asof(sinha_df1, flare_df1, on=col, tolerance=0.01 * sinha_df1[col].std(), direction="nearest")

        # df = pd.merge_asof(sinha_df1, flare_df1, left_on='TOTUSJH', right_on='SHRGT45', tolerance=2)

        print(df.columns)
        df = df[all_params].sort_values(by="index", ascending=True)
        # df = df[all_params]
        df.drop("index", axis=1, inplace=True)
        df.reset_index(inplace=True)
        df.drop("index", axis=1, inplace=True)
        df.rename(columns={k: v for k, v in zip(sinha_params, renamed_params)}, inplace=True)
        print(df.head(100).to_string())
        df.to_csv(f"{FLARE_DATA_DIRECTORY}sinha_mx_data_{col.lower()}.csv", index=False)

    df = get_median_df(cols)
    get_flare_info(df)



if __name__ == "__main__":
    main()