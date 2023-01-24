################################################################################
# Filename: time_series_goodness_of_fit.py
# Description:
################################################################################

# Custom Imports
import matplotlib.pyplot as plt
import pandas as pd

from source.utilities import *
from scipy.stats import ks_2samp, chi2, relfreq, chisquare
from scipy.signal import find_peaks
import json

# Experiment Name (No Acronyms)
experiment = "time_series_goodness_of_fit"
experiment_caption = experiment.title().replace("_", " ")

# ------------------------------------------------------------------------
# Place any results in the directory for the current experiment.
now_string, figure_directory, metrics_directory, other_directory = \
    build_experiment_directories(experiment)

# for index, row in flare_list.loc[flare_list["xray_class"] != "N"].iterrows():
#     nar = row["nar"]
#     time_end = row["time_start"]
#     time_start = time_end - timedelta(hours=24)
#
#     # Get all the other flares in this flare's AR.
#     # Then, determine if any of those flares coincide.
#     flares_in_ar = flare_list.loc[flare_list["nar"] == nar]
#     for index2, row2 in flares_in_ar.iterrows():
#         # Ignore the case when the flares are the same.
#         if index == index2:
#             break
#         time_start2 = row2["time_start"]
#         flares_coincide = time_start <= time_start2 <= time_end
#         if flares_coincide:
#             flare_list.loc[index2, "COINCIDENCE"] = [True]
#             flare_list.loc[index, "before_coincident_flare"] = [True]
#             break
#
# flare_list.to_csv(f"{FLARE_LIST_DIRECTORY}nbcmx_list.csv")
# exit(1)

flare_list = pd.read_csv(f"{FLARE_LIST_DIRECTORY}nbcmx_list.csv", header=0, parse_dates=["time_start", "time_peak", "time_end"], index_col="index")

x_list = flare_list.loc[flare_list["xray_class"] == "X"]
m_list = flare_list.loc[flare_list["xray_class"] == "M"]
b_list = flare_list.loc[flare_list["xray_class"] == "B"]
c_list = flare_list.loc[flare_list["xray_class"] == "C"]
n_list = flare_list.loc[flare_list["xray_class"] == "N"]

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
# b_data = None
# bc_data = None
# n_data = None
# nb_data = None

data = mx_data
class_list = m_list
c = "m"

def get_idealized_flare():
    for coincidence, label in zip([True, False], ["coincident", "noncoincident"]):
        for data, class_list, c in zip(
            [mx_data, mx_data, bc_data, b_data, n_data, mx_data, bc_data, nb_data],
            [x_list, m_list, c_list, b_list, n_list, mx_list, bc_list, nb_list],
            ["x", "m", "c", "b", "n", "mx", "bc", "nb"]
        ):
            class_list = class_list.loc[class_list["COINCIDENCE"] == coincidence]

            timepoint_sum_df = pd.DataFrame(np.zeros((122, len(FLARE_PROPERTIES))), columns=FLARE_PROPERTIES)
            timepoint_div_df = pd.DataFrame(np.zeros((122, len(FLARE_PROPERTIES))), columns=FLARE_PROPERTIES)

            i = 1
            for index, row in class_list.iterrows():
                print(f'{i}/{class_list.shape[0]}')
                df_1 = pd.DataFrame(columns=FLARE_PROPERTIES)
                for flare_property in FLARE_PROPERTIES:
                    df_1[flare_property] = np.zeros(122)
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
                    time_series_df = filter_data(time_series_df, nar)

                    if time_series_df.empty:
                        print(f"Skipping {flare_class} flare at {time_end} due to empty dataframe")
                        continue

                except IndexError:
                    print(f"Skipping {flare_class} flare at {time_end} due to IndexError")
                    continue
                except pd.errors.InvalidIndexError:
                    print(
                        f"Skipping {flare_class} flare at {time_end} due to InvalidIndexError")
                    continue

                for index, row in time_series_df.iterrows():
                    for param in FLARE_PROPERTIES:
                        timepoint_sum_df.loc[index, param] += row[param]
                        timepoint_div_df.loc[index, param] += 1

            (timepoint_sum_df.iloc[list(range(120))] / timepoint_div_df.iloc[list(range(120))]).to_csv(f"{other_directory}{label}/{c}_idealized_flare.csv")


def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def test():
    # for coincidence, label in zip([True, False],
    #                               ["coincident", "noncoincident"]):
    mx_df = pd.read_csv(f"{other_directory}{'mx'}_idealized_flare.csv", header=0)
    bc_df = pd.read_csv(f"{other_directory}{'bc'}_idealized_flare.csv", header=0)
    n_df = pd.read_csv(f"{other_directory}{'n'}_idealized_flare.csv", header=0)
    lag = 20
    threshold = 3.5
    influence = 0
    for param in FLARE_PROPERTIES:
        y = mx_df[param]
        y2 = bc_df[param]
        y3 = n_df[param]

        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        # Run algo with settings from above
        result = thresholding_algo(y, lag=lag, threshold=threshold,
                                   influence=influence)

        # Plot result
        ax[0, 0].set_title(f"MX {param}")
        ax[0, 0].plot(np.arange(1, len(y) + 1), y)

        ax[0, 0].plot(np.arange(1, len(y) + 1),
                   result["avgFilter"], color="orange", lw=2, alpha=0.75, ls="dotted",
                      label=f"Rolling Average, lag={lag}")

        ax[0, 0].plot(np.arange(1, len(y) + 1),
                   result["avgFilter"] + threshold * result["stdFilter"], ls="dashed",
                   color="green", lw=2, alpha=0.75, label=f"Threshold={threshold}")

        ax[0, 0].plot(np.arange(1, len(y) + 1),
                   result["avgFilter"] - threshold * result["stdFilter"], ls="dashed",
                   color="green", lw=2, alpha=0.75)

        ax[1, 0].step(np.arange(1, len(y) + 1), result["signals"], color="red", lw=2)
        ax[1, 0].set_ylim(-1.5, 1.5)

        result = thresholding_algo(y2, lag=lag, threshold=threshold,
                                   influence=influence)

        # Plot result
        ax[0, 1].set_title(f"BC {param}")
        ax[0, 1].plot(np.arange(1, len(y2) + 1), y2)

        ax[0, 1].plot(np.arange(1, len(y2) + 1),
                      result["avgFilter"], color="orange", lw=2, alpha=0.75,
                      ls="dotted",
                      label=f"Rolling Average, lag={lag}")

        ax[0, 1].plot(np.arange(1, len(y2) + 1),
                      result["avgFilter"] + threshold * result["stdFilter"],
                      ls="dashed",
                      color="green", lw=2, alpha=0.75, label=f"Threshold={threshold}")

        ax[0, 1].plot(np.arange(1, len(y2) + 1),
                      result["avgFilter"] - threshold * result["stdFilter"],
                      ls="dashed",
                      color="green", lw=2, alpha=0.75)


        ax[1, 1].step(np.arange(1, len(y2) + 1), result["signals"], color="red", lw=2)
        ax[1, 1].set_ylim(-1.5, 1.5)

        result = thresholding_algo(y3, lag=lag, threshold=threshold,
                                   influence=influence)

        # Plot result
        ax[0, 2].set_title(f"N {param}")
        ax[0, 2].plot(np.arange(1, len(y3) + 1), y3)

        ax[0, 2].plot(np.arange(1, len(y3) + 1),
                      result["avgFilter"], color="orange", lw=2, alpha=0.75,
                      ls="dotted",
                      label=f"Rolling Average, lag={lag}")

        ax[0, 2].plot(np.arange(1, len(y3) + 1),
                      result["avgFilter"] + threshold * result["stdFilter"],
                      ls="dashed",
                      color="green", lw=2, alpha=0.75,
                      label=f"Threshold={threshold}")

        ax[0, 2].plot(np.arange(1, len(y3) + 1),
                      result["avgFilter"] - threshold * result["stdFilter"],
                      ls="dashed",
                      color="green", lw=2, alpha=0.75)

        ax[1, 2].step(np.arange(1, len(y3) + 1), result["signals"], color="red",
                      lw=2)
        ax[1, 2].set_ylim(-1.5, 1.5)

        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[0, 2].legend()
        fig.suptitle("All")
        fig.savefig(f"{figure_directory}peak_detection/mx_bc_n_{param.lower()}.png")
        plt.show()


def idealized_flares_plot():
    for coincidence, label in zip([True, False], ["coincident", "noncoincident"]):
        flare_classes = ["n", "b", "c", "m", "x"]
        colors = ["grey", "blue", "green", "orange", "red"]
        fig, ax = plt.subplots(4, 5, figsize=(30, 20))
        t = np.arange(120)
        for flare_class, color in zip(flare_classes, colors):
            i, j = 0, 0
            idealized_df = pd.read_csv(f"{other_directory}{label}/{flare_class}_idealized_flare.csv", header=0)
            for param in FLARE_PROPERTIES:
                series = idealized_df[param]
                ax[i,j].plot(t, series, label=flare_class.upper(), color=color)
                ax[i,j].legend(loc="lower left")
                ax[i,j].set_xlabel("Timepoint")
                ax[i,j].set_ylabel("Value")
                ax[i,j].set_title(param)
                j += 1
                if j == 5:
                    i += 1
                    j = 0
        plt.tight_layout()
        fig.savefig(f"{figure_directory}idealized_flares/{label}/nbcmx_idealized_flares.png")
        plt.show()

        flare_classes = ["n", "bc", "mx"]
        colors = ["grey", "blue", "red"]
        fig, ax = plt.subplots(4, 5, figsize=(30, 20))
        t = np.arange(120)
        for flare_class, color in zip(flare_classes, colors):
            i, j = 0, 0
            idealized_df = pd.read_csv(
                f"{other_directory}{flare_class}_idealized_flare.csv", header=0)
            for param in FLARE_PROPERTIES:
                series = idealized_df[param]
                ax[i, j].plot(t, series, label=flare_class.upper(), color=color)
                ax[i, j].legend(loc="lower left")
                ax[i, j].set_xlabel("Timepoint")
                ax[i, j].set_ylabel("Value")
                ax[i, j].set_title(param)
                j += 1
                if j == 5:
                    i += 1
                    j = 0
        plt.tight_layout()
        fig.savefig(f"{figure_directory}idealized_flares/{label}/n_bc_mx_idealized_flares.png")
        plt.show()


def goodness_of_fit():
    for data, class_list, c in zip(
            [mx_data, mx_data, bc_data, b_data, n_data, mx_data, bc_data,
             nb_data],
            [x_list, m_list, c_list, b_list, n_list, mx_list, bc_list, nb_list],
            ["x", "m", "c", "b", "n", "mx", "bc", "nb"]
    ):
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


def goodness_of_fit2():
    mx_df = pd.read_csv(f"{other_directory}coincident/mx_idealized_flare.csv", header=0)
    bc_df = pd.read_csv(f"{other_directory}noncoincident/bc_idealized_flare.csv", header=0)
    stats, p_values, rejects = [], [], []
    for param in FLARE_PROPERTIES:
        # minimum = min(mx_df[param].min(),
        #               bc_df[param].min())
        # maximum = max(mx_df[param].max(),
        #               bc_df[param].max())
        # if minimum is np.nan or maximum is np.nan:
        #     continue
        # min_max_range = (minimum, maximum)
        print(mx_df.shape[0])
        mx_freq, _, _, _ = relfreq(mx_df[param], numbins=10,
                                            # defaultreallimits=min_max_range
                                   )
        bc_freq, _, _, _ = relfreq(bc_df[param],
                                                numbins=10,
                                                # defaultreallimits=min_max_range
                                   )
        print(mx_freq)
        print(bc_freq)
        stat, p_value = ks_2samp(mx_freq, bc_freq)
        stats.append(stat)
        p_values.append(p_value)
        rejects.append(p_value < 0.05)
        print(param, p_value)

    stats_df = pd.DataFrame({
        "ks_stat": stats,
        "p_value": p_values,
        "reject_95_conf": rejects,
    })
    stats_df.index = FLARE_PROPERTIES
    stats_df.to_csv(f"{metrics_directory}coincident_mx_noncoincident_bc_ks_test.csv")


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # print(mx_data)
    # get_idealized_flare()
    # idealized_flares_plot()

    # goodness_of_fit2()

    # test()

    goodness_of_fit()

if __name__ == "__main__":
    main()
