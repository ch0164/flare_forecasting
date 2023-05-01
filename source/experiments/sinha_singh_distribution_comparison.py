import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from source.common_imports import *
from source.constants import *
from source.utilities import *
# from distfit import distfit
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
    else:
        return "not applicable"


flare_list = pd.read_csv(f"{FLARE_LIST_DIRECTORY}nbcmx_list.csv", header=0, parse_dates=["time_start", "time_peak", "time_end"], index_col="index")

x_list = flare_list.loc[flare_list["xray_class"] == "X"]
m_list = flare_list.loc[flare_list["xray_class"] == "M"]
b_list = flare_list.loc[flare_list["xray_class"] == "B"]
c_list = flare_list.loc[flare_list["xray_class"] == "C"]
n_list = flare_list.loc[flare_list["xray_class"] == "N"]
n_list["magnitude"] = 0

mx_list = pd.concat([m_list, x_list])
nb_list = pd.concat([b_list, n_list])
bc_list = pd.concat([b_list, c_list])

mx_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_MX_ARs_and_errors.txt", header=0, delimiter=r"\s+")
mx_data["T_REC"] = mx_data["T_REC"].apply(parse_tai_string)
mx_data.set_index("T_REC", inplace=True)

b_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_OnlyB_ARs_and_errors.txt", header=0, delimiter=r"\s+")
b_data["T_REC"] = b_data["T_REC"].apply(parse_tai_string)
b_data.set_index("T_REC", inplace=True)

bc_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_ABC_ARs_and_errors.txt", header=0, delimiter=r"\s+")
bc_data["T_REC"] = bc_data["T_REC"].apply(parse_tai_string)
bc_data.set_index("T_REC", inplace=True)

n_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_No_flare_ARs_and_errors.txt", header=0, delimiter=r"\s+")
n_data["T_REC"] = n_data["T_REC"].apply(parse_tai_string)
n_data.set_index("T_REC", inplace=True)

nb_data = pd.concat([b_data, n_data])

def generate_sinha_timepoint_info() -> pd.DataFrame:
    # Get info dataframe for all flares in the dataset, then find coincidences.
    flare_classes = ["nbc", "mx"]
    all_info_df = pd.DataFrame()
    for flare_class in flare_classes:
        info_df = pd.read_csv(f"{FLARE_LIST_DIRECTORY}{flare_class}_list.txt")
        info_df.drop("index", axis=1, inplace=True)
        info_df["magnitude"] = info_df["xray_class"].apply(get_magnitude)
        info_df["xray_class"] = info_df["xray_class"].apply(get_flare_class)
        all_info_df = pd.concat([all_info_df, info_df])

    all_info_df["COINCIDENCE"] = False
    all_info_df["AR_class"] = all_info_df["xray_class"].apply(get_ar_class)
    all_info_df['time_peak'] = np.where(
        all_info_df['time_peak'] == "not applicable",
        all_info_df['time_start'], all_info_df['time_peak'])
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

    # all_info_df.to_csv(f"{FLARE_LIST_DIRECTORY}singh_nbcmx_info.csv")

    return all_info_df


def generate_sinha_timepoint_dataset(all_flare_df: pd.DataFrame=None) -> pd.DataFrame:
    # flare_classes = ["nbc", "mx"]
    flare_classes = ["x", "m", "c", "b", "n"]

    # nb_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_ABC_ARs_and_errors.txt",
    #                       delimiter=r"\s+", header=0)
    # nb_df = pd.concat([nb_df, pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_OnlyB_ARs_and_errors.txt", delimiter=r"\s+", header=0),
    #                    pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_No_flare_ARs_and_errors.txt", delimiter=r"\s+", header=0)])
    # nb_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}agu_bc_data.txt")
    # mx_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_MX_ARs_and_errors.txt",
    #                     delimiter=r"\s+", header=0)
    # for df in [nb_df, mx_df]:
    #     df["T_REC"] = df["T_REC"].apply(parse_tai_string)
    #     df.set_index("T_REC", inplace=True)
        # df.drop_duplicates("T_REC", inplace=True)

    for i in range(0, 25):
        all_flare_df = pd.DataFrame(columns=FLARE_PROPERTIES + ["expected_timepoint", "observed_timepoint", "time_difference"])
        for data, class_list, c in zip(
                [mx_data, mx_data, bc_data, b_data, n_data],
                [x_list, m_list, c_list, b_list, n_list],
                flare_classes
        ):
            all_flare_df = pd.concat([class_list, all_flare_df])
            for index, row in class_list.iterrows():
                print(f"{index}/{class_list.shape[0]}")
                dt = floor_minute(row["time_start"] - timedelta(hours=i))
                nar = row["nar"]

                temp = filter_data(data, nar)
                # temp = data
                ar_records = temp.loc[temp["NOAA_AR"] == nar]
                try:
                    record_index = ar_records.index[ar_records.index.get_loc(dt, method='nearest')]
                    # record_index = ar_records.index[ar_records.index.get_loc(dt)]
                except KeyError:
                    continue
                except pd.errors.InvalidIndexError:
                    continue
                record = ar_records.loc[record_index]
                try:
                    for flare_property in FLARE_PROPERTIES:
                        all_flare_df.loc[index, flare_property] = record[flare_property]
                except ValueError:
                    continue
                all_flare_df.loc[index, "expected_timepoint"] = dt
                all_flare_df.loc[index, "observed_timepoint"] = record_index
                all_flare_df.loc[index, "time_difference"] = abs(record_index - dt)
                # all_flare_df.loc[index, "expected_timepoint", "observed_timepoint", "time_difference"] = [dt, record_index, abs(record_index - dt)]


        all_flare_df.dropna(inplace=True)
        # all_flare_df.to_csv(f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_{i}h_nearest_timepoint_without_filter.csv")
        all_flare_df.to_csv(f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_{i}h_nearest_timepoint_with_filter.csv")
        # all_info_df.to_csv(f"{FLARE_DATA_DIRECTORY}singh_nbcmx_data.csv")

def timepoint_tss_plot():
    timepoint_range = range(0, 25)
    names = [
        "KNN",
        "LR",
        "RFC",
        "SVM",
        "DART",
        "LDA"
    ]
    coincidences = ["All Flares", "Coincidental Flares", "Noncoincidental Flares"]
    df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}temp.csv")
    for name in names:
        for coincidence in coincidences:
            name_df = df.loc[df["name"] == name]
            name_coin_df = name_df.loc[name_df["dataset"] == coincidence]
            tss = list(name_coin_df["tss"])
            print(tss)
            max_tss = max(tss)
            max_index_tss = tss.index(max_tss)
            plt.plot(range(len(tss)), tss)
            plt.axvline(max_index_tss, ls="dashed")
            plt.show()
            exit(1)

def year_1_report_bcmx_classification_comparison():
    names = [
        "KNN",
        "LR",
        "RFC",
        "SVM",
        "DART",
        "LDA"
    ]

    def get_tss(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        detection_rate = tp / float(tp + fn)
        false_alarm_rate = fp / float(fp + tn)
        tss = detection_rate - false_alarm_rate
        return tss

    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        LogisticRegression(C=1000, class_weight="balanced"),
        RandomForestClassifier(n_estimators=120),
        SVC(),
        lgb.LGBMClassifier(boosting_type="dart"),
        LinearDiscriminantAnalysis()
    ]

    # tss_df = pd.DataFrame(columns=["name", "tss", "timepoint", "dataset"])
    tss_df = pd.DataFrame(columns=["name", "tss", "std", "dataset"])
    # for df, label in zip([mean_df, timepoint_df, timeseries_df], ["0h-24h Parameter Mean",
    #                                                               "24h in Advance Timepoint",
    #                                                               "0h-24h Time Series, R_VALUE"]):
    timepoint_labels = ["default_timepoint_with_filter", "default_timepoint_without_filter", "nearest_timepoint_with_filter", "nearest_timepoint_without_filter"]
    for i in range(24, 25):
    # for timepoint_label in timepoint_labels:
    # for i in range(24, 0, -1):
    #     subdir = f"0h_{i}h"

        # timepoint_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_{24}h_{timepoint_label}.csv")
        # timepoint_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}timepoints/singh_nbcmx_data_{i}h_timepoint.csv")
        timepoint_df = pd.read_csv(r"C:\Users\youar\PycharmProjects\flare_forecasting\results\time_series_goodness_of_fit\other\mean_time_series\0h_24h_mean_time_series.csv")
        # timepoint_df = pd.read_csv(
        #     f"{other_directory}mean_datasets/nbcmx_{subdir}_mean_timepoint.csv")

        # time_series_df = pd.DataFrame()
        # for param in FLARE_PROPERTIES:
        #     if time_series_df.empty:
        #         time_series_df = pd.concat([
        #             time_series_df,
        #             pd.read_csv(f"{other_directory}{subdir}/{param}.csv", index_col=0)],
        #             axis=1)
        #     else:
        #         d = pd.read_csv(f"{other_directory}{subdir}/{param}.csv", index_col=0)
        #         d = d.drop(["FLARE_TYPE", "COINCIDENCE"], axis=1)
        #         time_series_df = pd.concat([time_series_df, d], axis=1)

        all_df = timepoint_df
        # all_df = all_df.loc[all_df["xray_class"] != "C"]
        coin_df = all_df.loc[all_df["COINCIDENCE"] == True]
        noncoin_df = all_df.loc[all_df["COINCIDENCE"] == False]
        for df, label in zip([all_df, coin_df, noncoin_df],
                             ["All Flares",
                              "Coincidental Flares",
                              "Noncoincidental Flares"]):
            print(label)
            if "xray_class" not in df.columns:
                xray_class = "FLARE_TYPE"
            else:
                xray_class = "xray_class"
            # df = df.loc[df[xray_class] != "N"]
            if "LABEL" not in df.columns:
                df["LABEL"] = df[xray_class].apply(get_ar_class)

            features = df.columns[3:-1]
            for name, clf in zip(names, classifiers):
                tss_list = []
                X = df[features]
                X = (X - X.min()) / (X.max() - X.min())
                y = df["LABEL"]
                for index in range(30):
                    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                        test_size=0.30,
                                                                        stratify=df[xray_class])

                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    tss = get_tss(y_test, y_pred)
                    tss_list.append(tss)
                # tss_df.loc[tss_df.shape[0]] = [name, np.mean(tss_list), i, label]
                tss_df.loc[tss_df.shape[0]] = [name, np.mean(tss_list), np.std(tss_list), label]
                print(tss_df)
    tss_df.to_csv(f"{other_directory}0h_24h_mean_time_series_classification.csv")

def main():
    # info_df = generate_sinha_timepoint_info()
    # df = pd.read_csv(FLARE_LIST_DIRECTORY + "agu_bcmx.csv")
    # for time in ["time_start", "time_end", "time_peak"]:
    #     info_df[time] = info_df[time].apply(parse_tai_string)
    # timepoint_tss_plot()
    # data_df = generate_sinha_timepoint_dataset()
    year_1_report_bcmx_classification_comparison()


if __name__ == "__main__":
    main()