################################################################################
# Filename: time_series_goodness_of_fit.py
# Description:
################################################################################

# Custom Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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

flare_list = pd.read_csv(f"{FLARE_LIST_DIRECTORY}nbcmx_list.csv", header=0, parse_dates=["time_start", "time_peak", "time_end"], index_col="index")

x_list = flare_list.loc[flare_list["xray_class"] == "X"]
m_list = flare_list.loc[flare_list["xray_class"] == "M"]
b_list = flare_list.loc[flare_list["xray_class"] == "B"]
c_list = flare_list.loc[flare_list["xray_class"] == "C"]
n_list = flare_list.loc[flare_list["xray_class"] == "N"]

mx_list = pd.concat([m_list, x_list])
nb_list = pd.concat([b_list, n_list])
bc_list = pd.concat([b_list, c_list])

# mx_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_MX_ARs_and_errors.txt", header=0, delimiter=r"\s+")
# mx_data["T_REC"] = mx_data["T_REC"].apply(parse_tai_string)
# mx_data.set_index("T_REC", inplace=True)
#
# b_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_OnlyB_ARs_and_errors.txt", header=0, delimiter=r"\s+")
# b_data["T_REC"] = b_data["T_REC"].apply(parse_tai_string)
# b_data.set_index("T_REC", inplace=True)
#
# bc_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_ABC_ARs_and_errors.txt", header=0, delimiter=r"\s+")
# bc_data["T_REC"] = bc_data["T_REC"].apply(parse_tai_string)
# bc_data.set_index("T_REC", inplace=True)
#
# n_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_No_flare_ARs_and_errors.txt", header=0, delimiter=r"\s+")
# n_data["T_REC"] = n_data["T_REC"].apply(parse_tai_string)
# n_data.set_index("T_REC", inplace=True)

# nb_data = pd.concat([b_data, n_data])
mx_data = None
b_data = None
bc_data = None
n_data = None
nb_data = None

data = mx_data
class_list = x_list
c = "x"

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

            print((timepoint_sum_df.iloc[list(range(120))] / timepoint_div_df.iloc[list(range(120))]))
            print((timepoint_sum_df.iloc[list(range(120))] / timepoint_div_df.iloc[list(range(120))]))
            # (timepoint_sum_df.iloc[list(range(120))] / timepoint_div_df.iloc[list(range(120))]).to_csv(f"{other_directory}{label}/{c}_idealized_flare.csv")


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


def individual_flares_plot():
    mx_df = pd.read_csv(f"{other_directory}{'x'}_idealized_flare.csv",
                        header=0)
    data = mx_data
    bc_df = pd.read_csv(f"{other_directory}{'bc'}_idealized_flare.csv",
                        header=0)
    n_df = pd.read_csv(f"{other_directory}{'n'}_idealized_flare.csv", header=0)
    lag = 20
    threshold = 3.5
    influence = 0

    param = "ABSNJZH"

    fig, ax = plt.subplots(1, figsize=(16, 9))

    m_list.reset_index(inplace=True)
    m_list.drop("index", axis=1, inplace=True)

    for index, row in x_list.iterrows():
        print(f'{index}/{class_list.shape[0]}')
        nar = row["nar"]
        flare_class = row["xray_class"]
        time_start = row["time_start"] - timedelta(hours=24)
        time_end = row["time_start"]
        try:
            nar_data = data.loc[data["NOAA_AR"] == nar]
            start_index = nar_data.index.get_indexer([time_start],
                                                     method='pad')
            end_index = nar_data.index.get_indexer([time_end],
                                                   method='backfill')
            start_index = nar_data.iloc[start_index].index[0]
            end_index = nar_data.iloc[end_index].index[0]
            time_series_df = nar_data.loc[
                             start_index:end_index].reset_index()
            time_series_df = filter_data(time_series_df, nar)

            if time_series_df.empty:
                print(
                    f"Skipping {flare_class} flare at {time_end} due to empty dataframe")
                continue

        except IndexError:
            print(
                f"Skipping {flare_class} flare at {time_end} due to IndexError")
            continue
        except pd.errors.InvalidIndexError:
            print(
                f"Skipping {flare_class} flare at {time_end} due to InvalidIndexError")
            continue

        ax.plot(np.arange(1, len(time_series_df[param]) + 1),
                      time_series_df[param], color="red", alpha=0.2)

    y = mx_df[param]
    # Run algo with settings from above
    result = thresholding_algo(y, lag=lag, threshold=threshold,
                               influence=influence)

    # Plot result
    ax.set_title(f"X {param} Individual Time Series vs. Idealized")
    ax.plot(np.arange(1, len(y) + 1),
                  result["avgFilter"], color="orange", lw=2, alpha=0.75,
                  ls="dotted",
                  label=f"Rolling Average, lag={lag}")

    ax.plot(np.arange(1, len(y) + 1),
                  result["avgFilter"] + threshold * result["stdFilter"],
                  ls="dashed",
                  color="green", lw=2, alpha=0.75,
                  label=f"Threshold={threshold}")

    ax.plot(np.arange(1, len(y) + 1),
                  result["avgFilter"] - threshold * result["stdFilter"],
                  ls="dashed",
                  color="green", lw=2, alpha=0.75)

    plt.savefig(f"{figure_directory}/x_{param.lower()}_individual_flares.png")





def time_series_classification():
    mx_df = pd.read_csv(f"{other_directory}{'m'}_idealized_flare.csv",
                        header=0)
    bc_df = pd.read_csv(f"{other_directory}{'bc'}_idealized_flare.csv",
                        header=0)
    n_df = pd.read_csv(f"{other_directory}{'n'}_idealized_flare.csv", header=0)
    lag = 20
    threshold = 3.5
    influence = 0

    for index, row in class_list.iterrows():
        print(f'{index}/{class_list.shape[0]}')
        nar = row["nar"]
        flare_class = row["xray_class"]
        time_start = row["time_start"] - timedelta(hours=24)
        time_end = row["time_start"]
        try:
            nar_data = data.loc[data["NOAA_AR"] == nar]
            start_index = nar_data.index.get_indexer([time_start],
                                                     method='pad')
            end_index = nar_data.index.get_indexer([time_end],
                                                   method='backfill')
            start_index = nar_data.iloc[start_index].index[0]
            end_index = nar_data.iloc[end_index].index[0]
            time_series_df = nar_data.loc[
                             start_index:end_index].reset_index()
            time_series_df = filter_data(time_series_df, nar)

            if time_series_df.empty:
                print(
                    f"Skipping {flare_class} flare at {time_end} due to empty dataframe")
                continue

        except IndexError:
            print(
                f"Skipping {flare_class} flare at {time_end} due to IndexError")
            continue
        except pd.errors.InvalidIndexError:
            print(
                f"Skipping {flare_class} flare at {time_end} due to InvalidIndexError")
            continue

        for param in FLARE_PROPERTIES:
            y = mx_df[param]
            fig, ax = plt.subplots(2, 3, figsize=(16, 9))
            # Run algo with settings from above
            result = thresholding_algo(y, lag=lag, threshold=threshold,
                                       influence=influence)

            # Plot result
            ax[0, 0].set_title(f"MX {param}")
            ax[0, 0].plot(np.arange(1, len(time_series_df[param]) + 1), time_series_df[param])

            ax[0, 0].plot(np.arange(1, len(y) + 1),
                          result["avgFilter"], color="orange", lw=2, alpha=0.75,
                          ls="dotted",
                          label=f"Rolling Average, lag={lag}")

            ax[0, 0].plot(np.arange(1, len(y) + 1),
                          result["avgFilter"] + threshold * result["stdFilter"],
                          ls="dashed",
                          color="green", lw=2, alpha=0.75,
                          label=f"Threshold={threshold}")

            ax[0, 0].plot(np.arange(1, len(y) + 1),
                          result["avgFilter"] - threshold * result["stdFilter"],
                          ls="dashed",
                          color="green", lw=2, alpha=0.75)
            plt.savefig(f"{figure_directory}peak_classification/{param}_{index}.png")
            break


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


def idealized_series_plot_2d():
    import plotly.express as px
    import plotly.graph_objs as go
    # for coincidence, label in zip([True, False], ["coincident", "noncoincident"]):
    flare_classes = ["n", "b", "c", "m", "x"]
    colors = ["grey", "blue", "green", "orange", "red"]
    t = np.arange(120)
    possible_param_pairs = [(a, b) for idx, a in enumerate(FLARE_PROPERTIES) for b in FLARE_PROPERTIES[idx + 1:]]
    flare_dfs = {flare_class: pd.read_csv(f"{other_directory}{flare_class}_idealized_flare.csv", header=0) for flare_class in flare_classes}
    for param1, param2 in possible_param_pairs:
        traces = []
        # minimum1, maximum1 = None, None
        # minimum2, maximum2 = None, None
        # for flare_class in flare_classes:
        #     if minimum1 is None or flare_dfs[flare_class][param1].min() < minimum1:
        #         minimum1 = flare_dfs[flare_class][param1].min()
        #     if minimum2 is None or flare_dfs[flare_class][param2].min() < minimum2:
        #         minimum2 = flare_dfs[flare_class][param2].min()
        #     if maximum1 is None or flare_dfs[flare_class][param1].max() > maximum1:
        #         maximum1 = flare_dfs[flare_class][param1].max()
        #     if maximum2 is None or flare_dfs[flare_class][param2].max() > maximum2:
        #         maximum2 = flare_dfs[flare_class][param2].max()
        for flare_class, color in zip(flare_classes, colors):
            df = flare_dfs[flare_class][[param1, param2]]
            df["index"] = df.index.tolist()
            df["z"] = [df.index.tolist()] * df.shape[0]
            # df[param1] = list((df[param1] - minimum1) / (maximum1 - minimum1))
            # df[param2] = list((df[param2] - minimum2) / (maximum2 - minimum2))
            trace = go.Scatter3d(x=df[param1], y=df[param2], z=df['index'],
                                # colorscale=[[i, color] for i in np.arange(0, 1.1, 0.1)],
                                # showscale=False,
                                 surfacecolor=color,
                                opacity=0.5,
                                name=flare_class.upper(),
                                showlegend=True,
                                 marker={
                                     "color": color
                                 },
                                 mode="lines")
            # print(df[["x", "y", "z"]])
            # trace = go.Surface(x=df[param1], y=df[param2], z=df['z'],
            #                     colorscale=[[i, color] for i in np.arange(0, 1.1, 0.1)],
            #                     showscale=False,
            #                     opacity=0.5,
            #                     name=flare_class.upper(),
            #                     showlegend=True,
            #                     )
            traces.append(trace)
            # fig.add_traces(traces)
            # print(traces)
        fig = go.Figure(data=traces)
        # fig.update_layout(xaxis_title=param1, yaxis_title=param2)
        fig.update_layout(
            title=f"{param1} vs. {param2}, Idealized Time-Series",
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title="Time",
            ),
        )
        fig.write_html(f"{other_directory}multivariate_flare_plots/idealized_flares/{param1.lower()}_vs_{param2.lower()}_raw.html")


def individual_time_series_plot_2d():
    import plotly.express as px
    import plotly.graph_objs as go
    # for coincidence, label in zip([True, False], ["coincident", "noncoincident"]):
    colors = ["grey", "blue", "green", "orange", "red"]
    possible_param_pairs = [(a, b) for idx, a in enumerate(FLARE_PROPERTIES) for b in FLARE_PROPERTIES[idx + 1:]]

    for param1, param2 in possible_param_pairs:
        traces = []
        for data, class_list, c, color in zip(
                [mx_data, mx_data, bc_data, b_data, n_data],
                [x_list, m_list, c_list, b_list, n_list],
                ["x", "m", "c", "b", "n"],
            ["red", "orange", "green", "blue", "grey"]
        ):
            for index, row in class_list.iterrows():
                nar = row["nar"]
                flare_class = row["xray_class"]
                time_start = row["time_start"] - timedelta(hours=24)
                time_end = row["time_start"]
                try:
                    nar_data = data.loc[data["NOAA_AR"] == nar]
                    start_index = nar_data.index.get_indexer([time_start],
                                                             method='pad')
                    end_index = nar_data.index.get_indexer([time_end],
                                                           method='backfill')
                    start_index = nar_data.iloc[start_index].index[0]
                    end_index = nar_data.iloc[end_index].index[0]
                    time_series_df = nar_data.loc[
                                     start_index:end_index].reset_index()
                    time_series_df = filter_data(time_series_df, nar)

                    if time_series_df.empty:

                        continue

                except IndexError:

                    continue
                except pd.errors.InvalidIndexError:

                    continue

                df = time_series_df[[param1, param2]]
                # df = (df-df.min())/(df.max()-df.min())
                df["index"] = df.index.tolist()
                trace = go.Scatter3d(x=df[param1], y=df[param2], z=df['index'],
                                    # colorscale=[[i, color] for i in np.arange(0, 1.1, 0.1)],
                                    # showscale=False,
                                     surfacecolor=color,
                                     marker={
                                         "color": color,
                                     },
                                    opacity=0.5,
                                     mode="lines")
                # print(df[["x", "y", "z"]])
                # trace = go.Surface(x=df[param1], y=df[param2], z=df['z'],
                #                     colorscale=[[i, color] for i in np.arange(0, 1.1, 0.1)],
                #                     showscale=False,
                #                     opacity=0.5,
                #                     name=flare_class.upper(),
                #                     showlegend=True,
                #                     )
                traces.append(trace)
                # fig.add_traces(traces)
                # print(traces)
        fig = go.Figure(data=traces)
        # fig.update_layout(xaxis_title=param1, yaxis_title=param2)
        fig.update_layout(
            title=f"{param1} vs. {param2}, Individual Time-Series",
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title="Time",
            ),
            showlegend=False,

        )
        fig.write_html(f"{other_directory}multivariate_flare_plots/individual_flares/all_{param1.lower()}_vs_{param2.lower()}.html")


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


def get_dataframe_of_vectors():
    # params = ["R_VALUE", "TOTUSJZ", "TOTUSJH"]
    # df = pd.DataFrame(columns=["FLARE_TYPE", "COINCIDENCE"] + [f"{param}_{i}" for i in range(1, 121) for param in params])
    flare_classes = ["x", "m", "c", "b", "n"]
    counts = {flare_class: 0 for flare_class in flare_classes}
    i = 0
    param_dfs = {param: pd.DataFrame(columns=["FLARE_TYPE", "COINCIDENCE"] + [f"{param}_{i}" for i in range(1, 121)]) for param in FLARE_PROPERTIES}
    for data, class_list, c in zip(
            [mx_data, mx_data, bc_data, b_data, n_data],
            [x_list, m_list, c_list, b_list, n_list],
            flare_classes
    ):
        for index, row in class_list.iterrows():
            i+=1
            print(f'{index}/{class_list.shape[0]}')
            nar = row["nar"]
            flare_class = row["xray_class"]
            coincidence = row["COINCIDENCE"]
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
                if time_series_df.shape[0] < 120:
                    continue
                time_series_df = time_series_df.iloc[0:120]
            except IndexError:
                print(
                    f"Skipping {flare_class} flare at {time_end} due to IndexError")
                continue
            except pd.errors.InvalidIndexError:
                print(
                    f"Skipping {flare_class} flare at {time_end} due to InvalidIndexError")
                continue

            for param in FLARE_PROPERTIES:
                param_values = time_series_df[param].tolist()
                param_dfs[param].loc[len(param_dfs[param])] = [flare_class, coincidence] + param_values

            # r_values = time_series_df["R_VALUE"].tolist()
            # totus_jz_values = time_series_df["TOTUSJZ"].tolist()
            # totus_jh_values = time_series_df["TOTUSJH"].tolist()
            # df.loc[len(df)] = [flare_class, coincidence] + r_values + totus_jz_values + totus_jh_values
            counts[flare_class.lower()] += 1
        print(counts)

    for param in FLARE_PROPERTIES:
        param_dfs[param].to_csv(f"{other_directory}{param.lower()}.csv")


def calc_tss(y_true=None, y_predict=None):
    """
    Calculates the true skill score for binary classification based on the output of the confusion
    table function
    """
    scores = confusion_matrix(y_true, y_predict).ravel()
    TN, FP, FN, TP = scores
    # print('TN={0}\tFP={1}\tFN={2}\tTP={3}'.format(TN, FP, FN, TP))
    tp_rate = TP / float(TP + FN) if TP > 0 else 0
    fp_rate = FP / float(FP + TN) if FP > 0 else 0

    return tp_rate - fp_rate


def get_ar_class(flare_class: str) -> int:
    if "M" in flare_class or "X" in flare_class:
        return 1
    else:
        return 0

def time_series_vector_classification():
    names = [
        "KNN",
        "RFC",
        "LR",
        "SVM",
        "LDA"
    ]

    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(n_estimators=120),
        LogisticRegression(C=1000),
        SVC(),
        LinearDiscriminantAnalysis()
    ]
    # Counts: {'x': 25, 'm': 434, 'c': 140, 'b': 784, 'n': 355}
    df = pd.read_csv(f"{other_directory}r_value_totusjz_totusjh.csv")
    df.drop("Unnamed: 0", axis=1, inplace=True)
    # df = df.loc[df["FLARE_TYPE"] != "C"]
    df["LABEL"] = df["FLARE_TYPE"].apply(get_ar_class)
    # feat_df = df.drop(["FLARE_TYPE", "COINCIDENCE", "LABEL"], axis=1)
    # feat_df = feat_df[[f"R_VALUE_{i}" for i in range(1, 121)] + [f"TOTUSJH_{i}" for i in range(1, 121)] + [f"TOTUSJZ_{i}" for i in range(1, 121)]]
    # r_value_mean_df = feat_df.iloc[:, [i + 45 for i in range(15)]].mean(axis=1)
    # totusjh_mean_df = feat_df.iloc[:, [i + 45 + 120 for i in range(15)]].mean(axis=1)
    # totusjz_mean_df = feat_df.iloc[:, [i + 45 + 240 for i in range(15)]].mean(axis=1)
    r_value_mean_df = df[[f"R_VALUE_{i}" for i in range(1, 121)]].mean(axis=1)
    totusjh_mean_df = df[[f"TOTUSJH_{i}" for i in range(1, 121)]].mean(axis=1)
    totusjz_mean_df = df[[f"TOTUSJZ_{i}" for i in range(1, 121)]].mean(axis=1)
    mean_df = pd.concat([r_value_mean_df, totusjh_mean_df, totusjz_mean_df], axis=1)
    mean_df.rename({i: key for i, key in enumerate(["R_VALUE", "TOTUSJH", "TOTUSJZ"])}, inplace=True, axis="columns")
    # feat_df = feat_df.iloc[:, [i + 45 for i in range(15)] + [i + 45 + 120 for i in range(15)] + [i + 45 + 240 for i in range(15)]]
    # df = pd.concat([df[["FLARE_TYPE", "COINCIDENCE", "LABEL"]], feat_df], axis=1)
    mean_df = pd.concat([df[["FLARE_TYPE", "COINCIDENCE", "LABEL"]], mean_df], axis=1)

    # df = df.loc[~df["FLARE_TYPE"].str.contains("C")]
    train_series, test_series = train_test_split(df, stratify=df["FLARE_TYPE"], test_size=0.25, random_state=1)
    train_mean, test_mean = train_test_split(mean_df, stratify=mean_df["FLARE_TYPE"], test_size=0.25, random_state=1)
    # train = train.loc[~df["FLARE_TYPE"].str.contains("C")]
    for flare_type in ["N", "B", "C", "M", "X"]:
        print(f"{flare_type}: {len(train_series.loc[train_series['FLARE_TYPE'] == flare_type])}")
    print("Total:", train_series.shape[0])
    print()
    for flare_type in ["N", "B", "C", "M", "X"]:
        print(f"{flare_type}: {len(test_series.loc[test_series['FLARE_TYPE'] == flare_type])}")
    print("Total:", test_series.shape[0])
    print()

    for feats in powerset(["R_VALUE", "TOTUSJH", "TOTUSJZ"]):
        print(feats)
        features = [f"{param}_{i}" for param in feats for i in range(1, 121)]
        # features = [f"{param}_{i}" for param in feats for i in range(46, 61)]
        train_X, train_y = train_series[features], train_series["LABEL"].values
        test_X, test_y = test_series[features], test_series["LABEL"].values

        train_X_mean, train_y_mean = train_mean[list(feats)], train_mean["LABEL"].values
        test_X_mean, test_y_mean = test_mean[list(feats)], test_mean["LABEL"].values
        # np.set_printoptions(threshold=sys.maxsize)

        # X, y = df[features], df["LABEL"]
        # print(X)
        # loo = LeaveOneOut()
        # loo.get_n_splits(X)
        # for name, clf in zip(names, classifiers):
        #     y_true = []
        #     y_pred = []
        #     for train_index, test_index in loo.split(X):
        #         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #         y_train, y_test = y[train_index], y[test_index]
        #         clf.fit(X_train, y_train)
        #         y_pred.append(clf.predict(X_test))
        #         y_true.append(y_test)
        #     tss = calc_tss(y_true, y_pred)
        #     print(f"{name}: {tss}")
        for clf, name in zip(classifiers, names):
            clf.fit(train_X, train_y)
            y_pred = clf.predict(test_X)
            tss_series = calc_tss(test_y, y_pred)
            clf.fit(train_X_mean, train_y_mean)
            y_pred_mean = clf.predict(test_X_mean)
            tss_mean = calc_tss(test_y_mean, y_pred_mean)
            print(f"{name} Time-series TSS: {tss_series}")
            print(f"{name} Mean Timepoint TSS: {tss_mean}")
            print()
        print()

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def time_series_vector_classification2():
    names = [
        "KNN",
        "RFC",
        "SVM",
    ]

    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(n_estimators=120),
        SVC(),
    ]
    # Counts: {'x': 25, 'm': 434, 'c': 140, 'b': 784, 'n': 355}
    df = pd.read_csv(f"{other_directory}r_value.csv")
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df = df.loc[df["FLARE_TYPE"] != "C"]
    df["LABEL"] = df["FLARE_TYPE"].apply(get_ar_class)
    r_value_mean_df = df[[f"R_VALUE_{i}" for i in range(1, 121)]].mean(axis=1)
    mean_df = pd.concat([df[["FLARE_TYPE", "COINCIDENCE", "LABEL"]], r_value_mean_df], axis=1)
    mean_df.rename({0: "R_VALUE"}, inplace=True, axis="columns")

    features = [f"R_VALUE_{i}" for i in range(1, 121)]
    mean_tss = {name: [] for name in names}
    series_tss = {name: [] for name in names}
    results_df = pd.DataFrame(columns=["name", "tss", "label"])
    for coincidence, coin_value in zip(["all", "coincident", "noncoincident"], [None, True, False]):
        if coin_value is not None:
            temp_series_df = df.loc[df["COINCIDENCE"] == coin_value]
            temp_mean_df = mean_df.loc[mean_df["COINCIDENCE"] == coin_value]
        else:
            temp_series_df = df.copy()
            temp_mean_df = mean_df.copy()
        for trial_index in range(30):
            print(f"{coincidence}, {trial_index}/30")
            train_series, test_series = train_test_split(temp_series_df, stratify=temp_series_df["LABEL"],
                                                         test_size=0.25)
            train_mean, test_mean = train_test_split(temp_mean_df,
                                                     stratify=temp_mean_df["LABEL"],
                                                     test_size=0.25)
            train_X, train_y = train_series[features], train_series["LABEL"].values
            test_X, test_y = test_series[features], test_series["LABEL"].values

            train_X_mean, train_y_mean = train_mean["R_VALUE"].values.reshape(-1, 1), train_mean[
                "LABEL"].values
            test_X_mean, test_y_mean = test_mean["R_VALUE"].values.reshape(-1, 1), test_mean["LABEL"].values

            for clf, name in zip(classifiers, names):
                clf.fit(train_X, train_y)
                y_pred = clf.predict(test_X)
                tss_series = calc_tss(test_y, y_pred)
                series_tss[name].append(tss_series)

                clf.fit(train_X_mean, train_y_mean)
                y_pred_mean = clf.predict(test_X_mean)
                tss_mean = calc_tss(test_y_mean, y_pred_mean)
                mean_tss[name].append(tss_mean)


        for name in names:
            tss_series = np.mean(series_tss[name])
            tss_mean = np.mean(mean_tss[name])
            results_df.loc[len(results_df)] = [name, tss_series, f"R_VALUE 24h Time-series, {coincidence}"]
            results_df.loc[len(results_df)] = [name, tss_mean, f"R_VALUE 24h Mean, {coincidence}"]

    results_df.to_csv(f"{metrics_directory}classification/r_value_coincidence_no_c.csv")


def time_series_vector_classification3():
    names = [
        "KNN",
        "LR",
        "RFC",
        "SVM",
    ]

    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        LogisticRegression(),
        RandomForestClassifier(n_estimators=120),
        SVC(),
    ]

    results_df = pd.DataFrame(columns=["name", "tss", "parameter", "coincidence", "label"])
    param_list = [param for param in powerset(FLARE_PROPERTIES) if (len(param) == 2)]

    coin_value = None
    coincidence = None
    # print(param_list)
    # exit(1)
    def get_features(params, low=1, high=121):
        features = []
        for param in list(params):
            features += [f"{param}_{i}" for i in range(low, high)]
        return features

    for params in param_list:
        # Counts: {'x': 25, 'm': 434, 'c': 140, 'b': 784, 'n': 355}
        if len(params) == 2:
            param1, param2 = params
            df = pd.concat([
                pd.read_csv(f"{other_directory}{param1}.csv"),
                pd.read_csv(f"{other_directory}{param2}.csv")
            ], axis=1)
        elif len(params) == 3:
            param1, param2, param3 = params
            df = pd.concat([
                pd.read_csv(f"{other_directory}{param1}.csv"),
                pd.read_csv(f"{other_directory}{param2}.csv"),
                pd.read_csv(f"{other_directory}{param3}.csv"),
            ], axis=1)
        params = list(params)
        params_name = "_".join(params)
        features = get_features(params)
        # df = pd.read_csv(f"{other_directory}{param}.csv")
        df.drop("Unnamed: 0", axis=1, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        # df = df.loc[df["FLARE_TYPE"] != "C"]
        # print(df)
        df["LABEL"] = df["FLARE_TYPE"].apply(get_ar_class)
        first12_df = df.drop(get_features(params, 61, 121), axis=1)
        center12_df = df.drop(get_features(params, 1, 31) + get_features(params, 91, 121), axis=1)
        last12_df = df.drop(get_features(params, 1, 61), axis=1)
        # param_mean_df = df[[f"{param}_{i}" for i in range(1, 121)]].mean(axis=1)
        # mean_df = pd.concat([df[["FLARE_TYPE", "COINCIDENCE", "LABEL"]], param_mean_df], axis=1)
        # mean_df.rename({0: param}, inplace=True, axis="columns")

        # mean_tss = {name: [] for name in names}
        series_tss = {name: [] for name in names}
        first12_tss = {name: [] for name in names}
        center12_tss = {name: [] for name in names}
        last12_tss = {name: [] for name in names}

        if coin_value is not None:
            temp_series_df = df.loc[df["COINCIDENCE"] == coin_value]
            # temp_mean_df = mean_df.loc[mean_df["COINCIDENCE"] == coin_value]
            temp_first12_df = first12_df.loc[first12_df["COINCIDENCE"] == coin_value]
            temp_center12_df = center12_df.loc[center12_df["COINCIDENCE"] == coin_value]
            temp_last12_df = last12_df.loc[last12_df["COINCIDENCE"] == coin_value]
        else:
            temp_series_df = df.copy()
            # temp_mean_df = mean_df.copy()
            temp_first12_df = first12_df.copy()
            temp_center12_df = center12_df.copy()
            temp_last12_df = last12_df.copy()
        for trial_index in range(30):
            print(f"{params}, {trial_index}/30")
            train_series, test_series = train_test_split(temp_series_df, stratify=temp_series_df["FLARE_TYPE"],
                                                         test_size=0.25)
            # train_mean, test_mean = train_test_split(temp_mean_df,
            #                                          stratify=temp_mean_df["FLARE_TYPE"],
            #                                          test_size=0.25)
            train_first12, test_first12 = train_test_split(temp_first12_df,
                                                         stratify=
                                                         temp_first12_df[
                                                             "FLARE_TYPE"],
                                                         test_size=0.25)
            train_center12, test_center12 = train_test_split(temp_center12_df,
                                                           stratify=
                                                           temp_center12_df[
                                                               "FLARE_TYPE"],
                                                           test_size=0.25)
            train_last12, test_last12 = train_test_split(temp_last12_df,
                                                           stratify=
                                                           temp_last12_df[
                                                               "FLARE_TYPE"],
                                                           test_size=0.25)
            train_X, train_y = train_series[features], train_series["LABEL"].values
            test_X, test_y = test_series[features], test_series["LABEL"].values

            train_first12_X, train_first12_y = train_first12.drop(["FLARE_TYPE", "COINCIDENCE", "LABEL"], axis=1), train_first12[
                "LABEL"].values
            test_first12_X, test_first12_y = test_first12.drop(["FLARE_TYPE", "COINCIDENCE", "LABEL"], axis=1), test_first12[
                "LABEL"].values

            train_center12_X, train_center12_y = train_center12.drop(["FLARE_TYPE", "COINCIDENCE", "LABEL"], axis=1), \
                                               train_center12[
                                                   "LABEL"].values
            test_center12_X, test_center12_y = test_center12.drop(["FLARE_TYPE", "COINCIDENCE", "LABEL"], axis=1), \
                                             test_center12[
                                                 "LABEL"].values

            train_last12_X, train_last12_y = train_last12.drop(["FLARE_TYPE", "COINCIDENCE", "LABEL"], axis=1), \
                                               train_last12[
                                                   "LABEL"].values
            test_last12_X, test_last12_y = test_last12.drop(["FLARE_TYPE", "COINCIDENCE", "LABEL"], axis=1), \
                                             test_last12[
                                                 "LABEL"].values

            # train_X_mean, train_y_mean = train_mean[params].values.reshape(-1, 1), train_mean[
            #     "LABEL"].values
            # test_X_mean, test_y_mean = test_mean[params].values.reshape(-1, 1), test_mean["LABEL"].values

            for clf, name in zip(classifiers, names):
                clf.fit(train_X, train_y)
                y_pred = clf.predict(test_X)
                tss_series = calc_tss(test_y, y_pred)
                series_tss[name].append(tss_series)

                clf.fit(train_first12_X, train_first12_y)
                y_pred = clf.predict(test_first12_X)
                tss_first12 = calc_tss(test_first12_y, y_pred)
                first12_tss[name].append(tss_first12)

                clf.fit(train_center12_X, train_center12_y)
                y_pred = clf.predict(test_center12_X)
                tss_center12 = calc_tss(test_center12_y, y_pred)
                center12_tss[name].append(tss_center12)

                clf.fit(train_last12_X, train_last12_y)
                y_pred = clf.predict(test_last12_X)
                tss_last12 = calc_tss(test_last12_y, y_pred)
                last12_tss[name].append(tss_last12)

                # clf.fit(train_X_mean, train_y_mean)
                # y_pred_mean = clf.predict(test_X_mean)
                # tss_mean = calc_tss(test_y_mean, y_pred_mean)
                # mean_tss[name].append(tss_mean)

                # print(f"\tfirst12: {tss_first12:.4f}, center12: {tss_center12:.4f}, last12: {tss_last12:.4f}")


        for name in names:
            tss_series = np.mean(series_tss[name])
            tss_first12 = np.mean(first12_tss[name])
            tss_center12 = np.mean(center12_tss[name])
            tss_last12 = np.mean(last12_tss[name])
            # tss_mean = np.mean(mean_tss[name])


            results_df.loc[len(results_df)] = [name, tss_series, params_name, coincidence, f"24h Time-series"]
            results_df.loc[len(results_df)] = [name, tss_first12, params_name,
                                               coincidence,
                                               f"0-12h Time-series"]
            results_df.loc[len(results_df)] = [name, tss_center12, params_name,
                                               coincidence,
                                               f"6h-18h Time-series"]
            results_df.loc[len(results_df)] = [name, tss_last12, params_name,
                                               coincidence,
                                               f"13-24h Time-series"]
            # results_df.loc[len(results_df)] = [name, tss_mean, params_name, coincidence, f"24h Mean"]

        results_df.to_csv(f"{metrics_directory}classification/all_params_pairs.csv", index=False)


def classification_plot():
    names = ["KNN", "RFC", "LR", "SVM"]
    import csv

    df = pd.read_csv(f"{metrics_directory}classification/all_params_coincidence.csv")
    def combine_label(row):
            return " ".join(row.values.astype(str))
    df["full_label"] = df[["coincidence", "label"]].apply(combine_label, axis=1)
    best_params = {}
    for coincidence in ["all", "coincident", "noncoincident"]:
        best_params_df = pd.DataFrame(columns=["name", "coincidence", "parameter", "label", "tss"])
        c_df = df.loc[df["coincidence"] == coincidence]
        for param in FLARE_PROPERTIES:
            d = c_df.loc[c_df["parameter"] == param]
            d.sort_values(by="tss", ascending=False, inplace=True)
            row = d.head(1)
            print(row)
            best_params_df = pd.concat([best_params_df, row[["name", "coincidence", "parameter", "label", "tss"]]])
            best_params_df.sort_values(by="tss", ascending=False, inplace=True)
            # best_params[param] = f"{row['label'].values[0]}: {row['tss'].values[0]}"

        # print(coincidence)
        # print(best_params)
        # coin_df = pd.DataFrame.from_dict(best_params)
        # print(coin_df)
        # print(best_params_df)
        # exit(1)
        best_params_df.to_csv(f"{metrics_directory}classification/{coincidence}_best_parameters.csv", index=False)

    # for data in ["all", "coincident", "noncoincident"]:
    #     tss_df = df.loc[df["coincidence"] == data]
    #     best_df = pd.DataFrame()
    #     def combine_label(row):
    #         return " ".join(row.values.astype(str))
    #
    #     tss_df["full_label"] = tss_df[["label", "parameter"]].apply(combine_label, axis=1)
    #     sns.set(rc={"figure.figsize": (19, 11)})
    #     for name in names:
    #         temp_df = tss_df.loc[tss_df["name"] == name]
    #         temp_df.sort_values(by="tss", ascending=False, inplace=True)
    #         best_df = pd.concat([best_df, temp_df.head(3)])
    #
    #     ax = sns.barplot(data=best_df, x="name", y="tss", hue="full_label")
    #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    #     ax.set_title(f"NBC vs. MX Classification Comparison, {data}")
    #     ax.set_ylim(bottom=0.0, top=1.0)
    #     ax.set_ylabel("TSS")
    #     ax.set_xlabel("Classifier")
    #     plt.tight_layout()
    #     plt.savefig(f"{figure_directory}classification/coincidence/{data} results.png")
    #     plt.show()
    #     plt.clf()

def generate_parallel_coordinates(coincidence, all_flares_df):
    fig, ax = plt.subplots(figsize=(19, 10))
    colors = ["blue", "green", "orange", "grey",  "red"]
    flare_classes = ["B", "C", "M", "N",  "X"]
    if coincidence == "coincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
    elif coincidence == "noncoincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]
    else:
        flares_df = all_flares_df
    # normalized_df = (properties_df - properties_df.mean()) / properties_df.std()
    properties_df = flares_df[FLARE_PROPERTIES]
    normalized_df = (properties_df - properties_df.min()) / (
                properties_df.max() - properties_df.min())
    normalized_df["xray_class"] = flares_df["xray_class"]

    print()
    normalized_df = normalized_df.sort_values(by="xray_class", ascending=True)
    print(normalized_df["xray_class"].to_string())
    parallel_coordinates(normalized_df, "xray_class", FLARE_PROPERTIES, ax, sort_labels=False,
                         color=colors, axvlines=True, axvlines_kwds={"color": "white"},
                         alpha=0.4)
    # ax.set_title(f"BCMX Flare Count, {coincidence.capitalize()} Flares,\n{study_caption}")
    # ax.set_xlabel("AR #")
    # ax.set_ylabel("# of Flares")
    # ax.legend(loc="upper left")
    # ax.set_xticks(np.arange(11600, 12301, step=100))
    # ax.set_yticks(np.arange(0, 26, step=5))
    # plt.gca().legend_.remove()
    plt.title(f"{coincidence.capitalize()}")
    plt.tight_layout()
    plt.savefig(f"{figure_directory}{coincidence}_parallel_coordinates.png")
    plt.show()



def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    # print(mx_data)
    # get_idealized_flare()
    # idealized_flares_plot()
    # classification_plot()
    # plot_parameter_tss()
    flare_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}singh_nbcmx_data.csv", index_col=False)
    flare_df = flare_df.loc[flare_df["xray_class"] != "A"]
    for coincidence in ["all", "coincident", "noncoincident"]:
        generate_parallel_coordinates(coincidence, flare_df)
        exit(1)
    # goodness_of_fit2()
    # time_series_vector_classification3()
    # test()
    # get_dataframe_of_vectors()
    # idealized_series_plot_2d()
    # individual_time_series_plot_2d()
    # individual_flares_plot()
    # exit(1)
    # time_series_classification()
    # classification_plot2()
    # resnet = keras.models.load_model(r"C:\Users\youar\PycharmProjects\flare_forecasting\resnet_mnist_digits\resnet_mnist_digits.hdf5")
    # resnet.summary()
    # goodness_of_fit()

if __name__ == "__main__":
    main()
