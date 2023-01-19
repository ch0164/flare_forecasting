################################################################################
# Filename: time_series_goodness_of_fit.py
# Description:
################################################################################

# Custom Imports
import matplotlib.pyplot as plt
import pandas as pd

from source.utilities import *
from scipy.stats import ks_2samp, chi2, relfreq, chisquare
import json

# Experiment Name (No Acronyms)
experiment = "time_series_goodness_of_fit"
experiment_caption = experiment.title().replace("_", " ")

# ------------------------------------------------------------------------
# Place any results in the directory for the current experiment.
now_string, figure_directory, metrics_directory, other_directory = \
    build_experiment_directories(experiment)

flare_list = pd.concat([pd.read_csv(f"{FLARE_LIST_DIRECTORY}nbc_list.txt", header=0, parse_dates=["time_start", "time_peak", "time_end"]),
                        pd.read_csv(f"{FLARE_LIST_DIRECTORY}mx_list.txt", header=0, parse_dates=["time_start", "time_peak", "time_end"])]).reset_index().drop(["level_0", "index"], axis=1)
# flare_list['time_start'] = pd.to_datetime(flare_list['time_start'], format='%Y-%m-%d %H:%M:%S')
# flare_list['time_peak'] = pd.to_datetime(flare_list['time_peak'], format='%Y-%m-%d %H:%M:%S')
# flare_list['time_end'] = pd.to_datetime(flare_list['time_end'], format='%Y-%m-%d %H:%M:%S')

flare_list["magnitude"] = flare_list["xray_class"].apply(get_magnitude)
flare_list["xray_class"] = flare_list["xray_class"].apply(classify_flare)

# x_list = flare_list.loc[flare_list["xray_class"] == "X"]
# m_list = flare_list.loc[flare_list["xray_class"] == "M"]
# b_list = flare_list.loc[flare_list["xray_class"] == "B"]
# c_list = flare_list.loc[flare_list["xray_class"] == "C"]
n_list = flare_list.loc[flare_list["xray_class"] == "N"]

mx_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_MX_ARs_and_errors.txt", header=0, delimiter=r"\s+")
mx_data["T_REC"] = mx_data["T_REC"].apply(parse_tai_string)
mx_data.set_index("T_REC", inplace=True)

# b_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_OnlyB_ARs_and_errors.txt", header=0, delimiter=r"\s+")
# b_data["T_REC"] = b_data["T_REC"].apply(parse_tai_string)
# b_data.set_index("T_REC", inplace=True)

# c_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}bc_data.txt", header=0, delimiter=r"\s+")
# c_data["T_REC"] = c_data["T_REC"].apply(parse_tai_string)
# c_data.set_index("T_REC", inplace=True)

n_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_No_flare_ARs_and_errors.txt", header=0, delimiter=r"\s+")
n_data["T_REC"] = n_data["T_REC"].apply(parse_tai_string)
n_data.set_index("T_REC", inplace=True)

data = n_data
class_list = n_list
c = "n"

def get_idealized_flare():
    timepoint_sum_df = pd.DataFrame(np.zeros((122, len(FLARE_PROPERTIES))), columns=FLARE_PROPERTIES)
    succeed = 0
    i = 1
    for index, row in class_list.iterrows():
        print(f'{i}/{class_list.shape[0]}')
        i += 1
        nar = row["nar"]
        flare_class = row["xray_class"]
        time_start = row["time_start"] - timedelta(hours=24)
        time_end = row["time_start"]
        try:
            nar_data = data.loc[data["NOAA_AR"] == nar]
            start_index = nar_data.index.get_indexer([time_start], method='pad')
            end_index = nar_data.index.get_indexer([time_end], method='backfill')
            start_index = nar_data.iloc[start_index].index[0]
            end_index = nar_data.iloc[end_index].index[0]
            time_series_df = nar_data.loc[start_index:end_index].reset_index()
        except IndexError:
            print(f"Skipping {flare_class} flare at {time_end} due to IndexError")
            continue
        except pd.errors.InvalidIndexError:
            print(
                f"Skipping {flare_class} flare at {time_end} due to InvalidIndexError")
            continue
        succeed += 1
        for index, row in time_series_df.iterrows():
            for param in FLARE_PROPERTIES:
                timepoint_sum_df.loc[index, param] += row[param]

    print(timepoint_sum_df)
    print((timepoint_sum_df / succeed).to_csv(f"{other_directory}{c}_idealized_flare.csv"))


def idealized_flares_plot():
    flare_classes = ["n", "b", "c", "m", "x"]
    fig, ax = plt.subplots(4, 5, figsize=(24, 20))
    for flare_class in flare_classes:
        i, j = 0, 0
        idealized_df = pd.read_csv(f"{other_directory}{flare_class}_idealized_flare.csv", header=0)
        for param in FLARE_PROPERTIES:
            ax[i,j].plot(range(idealized_df.shape[0]), idealized_df[param], label=flare_class.upper())
            ax[i,j].legend()
            ax[i,j].set_xlabel("Timepoint")
            ax[i,j].set_ylabel("Value")
            ax[i,j].set_title(param)
            ax[i,j].grid()
            j += 1
            if j == 5:
                i += 1
                j = 0
    plt.tight_layout()
    plt.show()
    exit(1)


def goodness_of_fit():
    succeed = 0
    i = 1
    idealized_flare_df = pd.read_csv(f"{other_directory}{c}_idealized_flare.csv", header=0)
    over_95_conf = {param: 0 for param in FLARE_PROPERTIES}
    p_values = {param: [] for param in FLARE_PROPERTIES}
    for index, row in class_list.iterrows():
        print(f'{i}/{class_list.shape[0]}')
        i += 1
        nar = row["nar"]
        flare_class = row["xray_class"]
        time_start = row["time_start"] - timedelta(hours=24)
        time_end = row["time_start"]
        try:
            nar_data = data.loc[data["NOAA_AR"] == nar]
            start_index = nar_data.index.get_indexer([time_start], method='pad')
            end_index = nar_data.index.get_indexer([time_end],
                                                   method='backfill')
            start_index = nar_data.iloc[start_index].index[0]
            end_index = nar_data.iloc[end_index].index[0]
            time_series_df = nar_data.loc[start_index:end_index].reset_index()
        except IndexError:
            print(
                f"Skipping {flare_class} flare at {time_end} due to IndexError")
            continue
        except pd.errors.InvalidIndexError:
            print(
                f"Skipping {flare_class} flare at {time_end} due to InvalidIndexError")
            continue
        succeed += 1
        for param in FLARE_PROPERTIES:
            minimum = min(time_series_df[param].min(), idealized_flare_df[param].min())
            maximum = max(time_series_df[param].max(), idealized_flare_df[param].max())
            if minimum is np.nan or maximum is np.nan:
                continue
            min_max_range = (minimum, maximum)
            time_series_freq, _, _, _ = relfreq(time_series_df[param], numbins=10,
                                          defaultreallimits=min_max_range)
            idealized_flare_freq, _, _, _ = relfreq(idealized_flare_df[param], numbins=10,
                                          defaultreallimits=min_max_range)
            stat, p_value = ks_2samp(time_series_freq, idealized_flare_freq)
            if p_value >= 0.05:
                over_95_conf[param] += 1
            p_values[param].append(p_value)
    over_95_conf_percentage = {param: over_95_conf[param] / succeed for param in FLARE_PROPERTIES}

    with open(f"{metrics_directory}{c}_ks_test.txt", "w") as fp:
        fp.write(f"Percentage of {c.upper()} Flares over 95% Confidence Level (Using KS 2-Sample Test Against Idealized Flare, 10 bins, {succeed} Flares)")
        json.dump(over_95_conf_percentage, fp, indent=4)
        fp.write(f"\nMean p-Value for each SHARP Parameter\n")
        for param in FLARE_PROPERTIES:
            fp.write(f"{param}: {np.mean(p_values[param])}\n")


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # print(mx_data)
    # get_idealized_flare()
    idealized_flares_plot()

    # goodness_of_fit()

if __name__ == "__main__":
    main()
