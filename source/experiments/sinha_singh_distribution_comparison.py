import pandas as pd
from matplotlib.colors import ListedColormap

from source.common_imports import *
from source.constants import *
from source.utilities import *
from distfit import distfit
from scipy.stats import ks_2samp
import numpy as np
from datetime import datetime as dt_obj
import datetime


# Place any results in the directory for the current experiment.
experiment = "sinha_singh_distribution_comparison"
now_string, figure_directory, metrics_directory, other_directory = \
    build_experiment_directories(experiment)

SINHA_PARAMETERS = [
    "TOTUSJH",
    "USFLUX",
    "TOTUSJZ",
    "R_VALUE",
    "TOTPOT",
    "AREA_ACR",
    "SAVNCPP",
    "ABSNJZH",
    "MEANPOT",
    "SHRGT45",
]


def get_magnitude(xray_class: str) -> float:
    if len(xray_class) >= 4:
        return float(xray_class[1:])
    else:
        return 1.0


def get_flare_class(xray_class: str) -> str:
    flare_classes = ["N", "A", "B", "C", "M", "X"]
    for flare_class in flare_classes:
        if flare_class in xray_class:
            return flare_class
    else:
        return ""


def get_ar_class(flare_class: str) -> int:
    if "M" in flare_class or "X" in flare_class:
        return 1
    else:
        return 0


def parse_tai_string(tstr: str):
    if "not applicable" in tstr:
        return "not applicable"
    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])
    return dt_obj(year, month, day, hour, minute)


def floor_minute(time, cadence=12):
    if not isinstance(time, str):
        return time - datetime.timedelta(minutes=time.minute % cadence)


def generate_sinha_timepoint_info() -> pd.DataFrame:
    # Get info dataframe for all flares in the dataset, then find coincidences.
    flare_classes = ["nb", "mx"]
    all_info_df = pd.DataFrame()
    for flare_class in flare_classes:
        info_df = pd.read_csv(f"{FLARE_LIST_DIRECTORY}{flare_class}_list.txt")
        info_df.drop("index", axis=1, inplace=True)
        info_df["magnitude"] = info_df["xray_class"].apply(get_magnitude)
        info_df["xray_class"] = info_df["xray_class"].apply(get_flare_class)
        all_info_df = pd.concat([all_info_df, info_df])

    all_info_df["COINCIDENCE"] = False
    all_info_df["AR_class"] = all_info_df["xray_class"].apply(get_ar_class)
    for time in ["time_start", "time_end", "time_peak"]:
        all_info_df[time] = all_info_df[time].apply(parse_tai_string)
    all_info_df.reset_index(inplace=True)
    all_info_df.drop("index", axis=1, inplace=True)
    for index, row in all_info_df.iterrows():
        nar = row["nar"]
        time_end = row["time_start"]
        time_start = time_end - timedelta(hours=24)

        # Get all the other flares in this flare's AR.
        # Then, determine if any of those flares coincide.
        flares_in_ar = all_info_df.loc[all_info_df["nar"] == nar]
        for index2, row2 in flares_in_ar.iterrows():
            # Ignore the case when the flares are the same.
            if index == index2:
                break
            time_start2 = row2["time_start"]
            flares_coincide = time_start <= time_start2 <= time_end
            if flares_coincide:
                all_info_df.loc[index, "COINCIDENCE"] = [True]
                break
    # all_info_df.to_csv(f"{FLARE_LIST_DIRECTORY}singh_nbmx_info.csv")

    return all_info_df


def generate_sinha_timepoint_dataset(all_info_df: pd.DataFrame) -> pd.DataFrame:
    flare_classes = ["nb", "mx"]
    for time in ["time_start", "time_end", "time_peak"]:
        all_info_df[time] = all_info_df[time].apply(floor_minute)

    nb_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}nb_data.txt",
                          delimiter=r"\s+", header=0)
    mx_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}mx_data.txt",
                        delimiter=r"\s+", header=0)
    for df in [nb_df, mx_df]:
        df["T_REC"] = df["T_REC"].apply(parse_tai_string)
        df.set_index("T_REC", inplace=True)
        # df.drop_duplicates("T_REC", inplace=True)

    for index, row in all_info_df.iterrows():
        print(f"{index}/{all_info_df.shape[0]}")
        dt = row["time_peak"] - timedelta(hours=24)
        flare_class = row["xray_class"]
        nar = row["nar"]

        if flare_class in "NAB":
            data_df = nb_df
        else:
            data_df = mx_df

        ar_records = data_df.loc[data_df["NOAA_AR"] == nar]
        try:
            record_index = ar_records.index[ar_records.index.get_loc(dt, method='nearest')]
        except KeyError:
            continue
        except pd.errors.InvalidIndexError:
            continue
        record = ar_records.loc[record_index]
        for flare_property in FLARE_PROPERTIES:
            all_info_df.loc[index, flare_property] = record[flare_property]

    all_info_df.dropna(inplace=True)
    all_info_df.to_csv(f"{FLARE_DATA_DIRECTORY}singh_nbmx_data.csv")



def main():
    info_df = generate_sinha_timepoint_info()
    data_df = generate_sinha_timepoint_dataset(info_df)


if __name__ == "__main__":
    main()