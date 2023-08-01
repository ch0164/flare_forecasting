################################################################################
# Filename: time_series_goodness_of_fit.py
# Description:
################################################################################

# Custom Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
import seaborn as sns

from source.utilities import *
from scipy.stats import ks_2samp, chi2, relfreq, chisquare, combine_pvalues
from scipy.signal import find_peaks
import json
import lightgbm as lgb
from scipy.stats import ranksums

# Experiment Name (No Acronyms)
experiment = "time_series_goodness_of_fit"
experiment_caption = experiment.title().replace("_", " ")

# ------------------------------------------------------------------------
# Place any results in the directory for the current experiment.
now_string, figure_directory, metrics_directory, other_directory = \
    build_experiment_directories(experiment)

SMALL_SIZE = 15
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

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
nbc_list = pd.concat([n_list, bc_list])

mx_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_MX_ARs_and_errors.txt", header=0, delimiter=r"\s+")
mx_data["T_REC"] = mx_data["T_REC"].apply(parse_tai_string)
mx_data.set_index("T_REC", inplace=True)
#
b_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_OnlyB_ARs_and_errors.txt", header=0, delimiter=r"\s+")
b_data["T_REC"] = b_data["T_REC"].apply(parse_tai_string)
b_data.set_index("T_REC", inplace=True)
#
bc_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_ABC_ARs_and_errors.txt", header=0, delimiter=r"\s+")
bc_data["T_REC"] = bc_data["T_REC"].apply(parse_tai_string)
bc_data.set_index("T_REC", inplace=True)
#
n_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_No_flare_ARs_and_errors.txt", header=0, delimiter=r"\s+")
n_data["T_REC"] = n_data["T_REC"].apply(parse_tai_string)
n_data.set_index("T_REC", inplace=True)
#
nb_data = pd.concat([b_data, n_data])
# mx_data = None
# b_data = None
# bc_data = None
# n_data = None
# nb_data = None

data = mx_data
class_list = x_list
c = "x"

def get_idealized_flare():
    for coincidence, label in zip([None, True, False], ["all", "coincident", "noncoincident"]):
        for data, class_list, c in zip(
            [mx_data, mx_data, bc_data, b_data, n_data],
            [x_list, m_list, c_list, b_list, n_list],
            ["x", "m", "c", "b", "n"]
        ):
            if coincidence != None:
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
                    # time_series_df = filter_data(time_series_df, nar)

                    if time_series_df.shape[0] < 120:
                        continue

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

                print(time_series_df)
                for index, row in time_series_df.iterrows():
                    for param in FLARE_PROPERTIES:
                        timepoint_sum_df.loc[index, param] += row[param]
                        timepoint_div_df.loc[index, param] += 1

            print((timepoint_sum_df.iloc[list(range(120))] / timepoint_div_df.iloc[list(range(120))]))
            (timepoint_sum_df.iloc[list(range(120))] / timepoint_div_df.iloc[list(range(120))]).to_csv(f"{other_directory}{label}/{c}_idealized_flare_unfilter_complete.csv")


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

    fig, ax = plt.subplots(1, figsize=(10, 7))

    m_list.reset_index(inplace=True)
    m_list.drop("index", axis=1, inplace=True)

    for index, row in x_list.iterrows():
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
                      time_series_df[param], color="red")

    y = mx_df[param]
    # Run algo with settings from above
    # result = thresholding_algo(y, lag=lag, threshold=threshold,
    #                            influence=influence)
    #
    # # Plot result
    ax.set_title(f"{param}, X Flares, Individual Time-Series vs. Idealized")
    ax.plot(np.arange(1, len(y) + 1), y, color="black", lw=2) # ls="dotted"
    line1 = Line2D([0], [0], label="Idealized X Flare", color='k', lw=2)  # ls="dotted"
    line2 = Line2D([0], [0], label="Individual X Flare", color='r')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([line1, line2])
    plt.legend(handles=handles, loc="lower left")
    # line = Line2D([0], [0], label='manual line', color='k')
    #
    # ax.plot(np.arange(1, len(y) + 1),
    #               result["avgFilter"] + threshold * result["stdFilter"],
    #               ls="dashed",
    #               color="green", lw=2, alpha=0.75,
    #               label=f"Threshold={threshold}")
    #
    # ax.plot(np.arange(1, len(y) + 1),
    #               result["avgFilter"] - threshold * result["stdFilter"],
    #               ls="dashed",
    #               color="green", lw=2, alpha=0.75)

    plt.savefig(f"{figure_directory}/x_{param.lower()}_individual_flares.eps")
    plt.show()





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
    for label in ["all", "coincident", "noncoincident"]:
        flare_classes = ["n", "b", "c", "m", "x"]
        colors = ["grey", "blue", "green", "orange", "red"]
        fig, ax = plt.subplots(4, 5, figsize=(45, 30))
        t = np.arange(120)
        for flare_class, color in zip(flare_classes, colors):
            i, j = 0, 0
            # if label == "all":
            #     idealized_df = pd.read_csv(f"{other_directory}{flare_class}_idealized_flare.csv", header=0)
            # else:
            idealized_df = pd.read_csv(f"{other_directory}{label}/{flare_class}_idealized_flare_unfilter_complete.csv", header=0)
            for param in FLARE_PROPERTIES:
                series = idealized_df[param]
                ax[i,j].plot(t, series, label=flare_class.upper(), color=color)
                ax[i,j].legend(loc="lower left", fontsize=MEDIUM_SIZE)
                ax[i,j].set_xlabel("Timepoint", fontsize=MEDIUM_SIZE)
                ax[i,j].set_ylabel("Value", fontsize=MEDIUM_SIZE)
                ax[i,j].set_title(param, fontsize=BIGGER_SIZE)
                ax[i,j].tick_params(axis='x', labelsize=SMALL_SIZE)
                ax[i,j].tick_params(axis='y', labelsize=SMALL_SIZE)
                exp = ax[i,j].yaxis.get_offset_text()
                exp.set_size(SMALL_SIZE)
                # plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
                # plt.rc('axes',
                #        titlesize=BIGGER_SIZE)  # fontsize of the axes title
                # plt.rc('axes',
                #        labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
                # plt.rc('xtick',
                #        labelsize=BIGGER_SIZE)  # fontsize of the tick labels
                # plt.rc('ytick',
                #        labelsize=BIGGER_SIZE)  # fontsize of the tick labels
                # plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
                # plt.rc('figure',
                #        titlesize=BIGGER_SIZE)  # fontsize of the figure title
                j += 1
                if j == 5:
                    i += 1
                    j = 0
        plt.tight_layout()


        fig.savefig(f"{figure_directory}idealized_flares/{label}/nbcmx_idealized_flares_unfilter_complete.eps")
        plt.show()

        # flare_classes = ["n", "bc", "mx"]
        # colors = ["grey", "blue", "red"]
        # fig, ax = plt.subplots(4, 5, figsize=(30, 20))
        # t = np.arange(120)
        # for flare_class, color in zip(flare_classes, colors):
        #     i, j = 0, 0
        #     idealized_df = pd.read_csv(
        #         f"{other_directory}{flare_class}_idealized_flare.csv", header=0)
        #     for param in FLARE_PROPERTIES:
        #         series = idealized_df[param]
        #         ax[i, j].plot(t, series, label=flare_class.upper(), color=color)
        #         ax[i, j].legend(loc="lower left")
        #         ax[i, j].set_xlabel("Timepoint")
        #         ax[i, j].set_ylabel("Value")
        #         ax[i, j].set_title(param)
        #         j += 1
        #         if j == 5:
        #             i += 1
        #             j = 0
        # plt.tight_layout()
        # fig.savefig(f"{figure_directory}idealized_flares/{label}/n_bc_mx_idealized_flares.png")
        # plt.show()


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
    missing_singles = {flare_class: 0 for flare_class in flare_classes}
    missing_pairs = {flare_class: 0 for flare_class in flare_classes}
    i = 0
    param_dfs = {param: pd.DataFrame(columns=["FLARE_TYPE", "COINCIDENCE"] + [f"{param}_{i}" for i in range(1, 119)]) for param in FLARE_PROPERTIES}  # 121
    x = 0
    for data, class_list, c in zip(
            [mx_data, mx_data, bc_data, b_data, n_data],
            [x_list, m_list, c_list, b_list, n_list],
            flare_classes
    ):
        for index, row in class_list.iterrows():
            i+=1
            # print(f'{index}/{class_list.shape[0]}')
            nar = row["nar"]
            flare_class = row["xray_class"]
            coincidence = row["COINCIDENCE"]
            time_start = row["time_start"] - timedelta(hours=24)
            time_end = row["time_start"]
            try:
                nar_data = data.loc[data["NOAA_AR"] == nar]
                start_index = nar_data.index.get_indexer([time_start], method='nearest')
                end_index = nar_data.index.get_indexer([time_end],
                                                       method='nearest')
                start_index = nar_data.iloc[start_index].index[0]
                end_index = nar_data.iloc[end_index].index[0]
                time_series_df = nar_data.loc[start_index:end_index]

                time_series_df["timestamp"] = time_series_df.index
                time_series_df['minutes'] = time_series_df['timestamp'].diff()
                # isolated_single_dropout_count = time_series_df.loc[
                #     time_series_df["minutes"] == pd.Timedelta(minutes=24)]
                isolated_pair_dropout_count = time_series_df.loc[
                    time_series_df["minutes"] == pd.Timedelta(minutes=36)]

                if isolated_pair_dropout_count.shape[0] == 1:
                # if time_series_df.shape[0] == 119:
                #     print(end_index)
                #     print(isolated_single_dropout_count)
                    # if missing_singles[flare_class.lower()] == 0:
                    #     time_series_df = time_series_df.reindex(pd.date_range(start_index, end_index, freq="12min",
                    #                       inclusive="left"), fill_value=np.NaN)
                    #     time_series_df["number"] = [i for i in range(1, time_series_df.shape[0] + 1)]
                    #     print(time_series_df.to_string())
                    #     exit(1)
                    # missing_singles[flare_class.lower()] += 1
                    # if missing_pairs[flare_class.lower()] == 7:
                    #     time_series_df = time_series_df.reindex(pd.date_range(start_index, end_index, freq="12min",
                    #                       inclusive="left"), fill_value=np.NaN)
                    #     time_series_df["number"] = [i for i in range(1, time_series_df.shape[0] + 1)]
                    #     print(time_series_df.to_string())
                    #     exit(1)
                    missing_pairs[flare_class.lower()] += 1
                    # continue


                # time_series_df = filter_data(time_series_df, nar)

                # if time_series_df.shape[0] < 118:
                #     time_series_df = time_series_df.reindex(
                #         pd.date_range(start=start_index, end=end_index,
                #                       freq='12T'))
                #     for param in FLARE_PROPERTIES:
                #         time_series_df[param] = time_series_df[param].interpolate(method="time")

                if time_series_df.shape[0] < 118:  # 120
                    continue
                # if time_series_df.shape[0] == 119:
                # if time_series_df["ABSNJZH"][0] == 251.86:
                #     # if x <= -1:
                #     #     x += 1
                #     #     continue
                #     print(time_series_df)
                #     print(pd.date_range(start_index, end_index, freq="12min", inclusive="left"))
                #     time_series_df = time_series_df.reindex(pd.date_range(start_index, end_index, freq="12min", inclusive="left"), fill_value=np.NaN)
                #     time_series_df["number"] = [i for i in range(1, 121)]
                #     # time_series_df.fillna(inplace=True)
                #     print(time_series_df.to_string())
                #     exit(1)
                time_series_df = time_series_df.reset_index()
                time_series_df = time_series_df.iloc[0:118]  # 120
            except IndexError:
                print(
                    f"Skipping {flare_class} flare at {time_end} due to IndexError")
                continue
            except pd.errors.InvalidIndexError:
                print(
                    f"Skipping {flare_class} flare at {time_end} due to InvalidIndexError")
                continue

            for param in FLARE_PROPERTIES:
                # try:
                param_values = time_series_df[param].tolist()
                param_dfs[param].loc[len(param_dfs[param])] = [flare_class, coincidence] + param_values
                # except ValueError:
                #     time_series_df.reset_index(inplace=True)
                #     print(time_series_df.to_string())
                #     print(param_values)
                #     print(len(param_values))
                #     exit(1)

            # r_values = time_series_df["R_VALUE"].tolist()
            # totus_jz_values = time_series_df["TOTUSJZ"].tolist()
            # totus_jh_values = time_series_df["TOTUSJH"].tolist()
            # df.loc[len(df)] = [flare_class, coincidence] + r_values + totus_jz_values + totus_jh_values
            counts[flare_class.lower()] += 1

        print(counts)
        print(missing_pairs)

    for param in FLARE_PROPERTIES:
        param_dfs[param].to_csv(f"{other_directory}0h_24h/{param.lower()}_missing_pair.csv")

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
    fig, ax = plt.subplots(figsize=(38, 20))
    colors = ["blue", "green", "orange", "grey",  "red"]
    # colors = ["green"]
    flare_classes = ["B", "C", "M", "N",  "X"]
    properties_df = all_flares_df[FLARE_PROPERTIES]
    normalized_df = (properties_df - properties_df.min()) / (
            properties_df.max() - properties_df.min())
    normalized_df["xray_class"] = all_flares_df["xray_class"]
    normalized_df["COINCIDENCE"] = all_flares_df["COINCIDENCE"]
    normalized_df = normalized_df.sort_values(by="xray_class", ascending=True)
    if coincidence == "coincident":
        normalized_df = normalized_df.loc[normalized_df["COINCIDENCE"] == True]
        colors = ["blue", "green", "orange", "red"]
        # colors = ["green"]
    elif coincidence == "noncoincident":
        normalized_df = normalized_df.loc[normalized_df["COINCIDENCE"] == False]
    else:
        normalized_df = normalized_df
    # normalized_df = (properties_df - properties_df.mean()) / properties_df.std()


    print()

    # normalized_df = normalized_df.loc[normalized_df["xray_class"] == "C"]
    props = ["R_VALUE", "TOTUSJH", "TOTUSJZ"]
    parallel_coordinates(normalized_df, "xray_class", props, ax, sort_labels=False,
                         color=colors, axvlines=True, axvlines_kwds={"color": "white"},
                         alpha=0.4)
    # ax.set_title(f"BCMX Flare Count, {coincidence.capitalize()} Flares,\n{study_caption}")
    # ax.set_xlabel("AR #")
    # ax.set_ylabel("# of Flares")
    # ax.legend(loc="upper left")
    # ax.set_xticks(np.arange(11600, 12301, step=100))
    # ax.set_yticks(np.arange(0, 26, step=5))
    # plt.gca().legend_.remove()
    plt.ylim(0.0, 1.0)
    plt.title(f"{coincidence.capitalize()}", fontsize=BIGGER_SIZE)
    plt.tick_params(axis="y", labelsize=MEDIUM_SIZE)
    plt.tick_params(axis="x", labelsize=MEDIUM_SIZE)
    plt.legend(prop={'size': MEDIUM_SIZE})
    plt.tight_layout()
    plt.savefig(f"{figure_directory}{coincidence}_parallel_coordinates_only_three_params.eps")
    plt.show()


def get_features(params, low=1, high=121, keep=120, randomize=False):
    features = []
    for param in list(params):
        f = [f"{param}_{i}" for i in range(low, high)]
        if randomize:
            np.random.shuffle(f)
        features += f[:keep]
    return features

def time_series_vector_classification4(dir=""):
    names = [
        "KNN",
        "LDA",
        "DART",
        "RFC",
        "SVM",
    ]

    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        LinearDiscriminantAnalysis(),
        lgb.LGBMClassifier(boosting_type="dart"),
        RandomForestClassifier(n_estimators=120),
        SVC(),
    ]

    results_df = pd.DataFrame(columns=["name", "tss_mean", "tss_std", "label"])

    data_df = pd.DataFrame()
    for param in FLARE_PROPERTIES:
        if data_df.empty:
            data_df = pd.concat([
                data_df,
                pd.read_csv(f"{other_directory}{dir}{param}.csv", index_col=0)],
                axis=1)
        else:
            d = pd.read_csv(f"{other_directory}{dir}{param}.csv", index_col=0)
            d = d[get_features([param])]
            data_df = pd.concat([data_df, d], axis=1)

    data_df["LABEL"] = data_df["FLARE_TYPE"].apply(get_ar_class)

    complete_features = get_features(FLARE_PROPERTIES)
    randomized_features = get_features(FLARE_PROPERTIES, randomize=True)
    unrandomized_series_tss = {name: [] for name in names}
    randomized_series_tss = {name: [] for name in names}
    df = data_df[["FLARE_TYPE", "COINCIDENCE", "LABEL"] + complete_features]

    for clf, name in zip(classifiers, names):
        for index in range(30):
            print(f"{name} {index}/30")
            train_df, test_df = train_test_split(df, test_size=0.3,
                                                 stratify=data_df["FLARE_TYPE"])
            train_X, unrandomized_test_X = train_df[complete_features], test_df[complete_features]
            train_y, test_y = train_df["LABEL"].values, test_df["LABEL"].values
            randomized_test_X = test_df[randomized_features]

            clf.fit(train_X, train_y)
            unrandomized_y_pred = clf.predict(unrandomized_test_X)
            randomized_y_pred = clf.predict(randomized_test_X)
            unrandomized_series_tss[name].append(calc_tss(test_y, unrandomized_y_pred))
            randomized_series_tss[name].append(calc_tss(test_y, randomized_y_pred))
        results_df.loc[results_df.shape[0]] = [name, np.mean(unrandomized_series_tss[name]), np.std(unrandomized_series_tss[name]), "0h-24h time-series"]
        results_df.loc[results_df.shape[0]] = [name, np.mean(randomized_series_tss[name]), np.std(randomized_series_tss[name]), "Randomized 60 time-points"]
        print(results_df)
        results_df.to_csv(f"{metrics_directory}randomized_time_series_comparison.csv")


def generate_mean_dataset():
    for hour in range(0, 1):
    # for hour in range(0, 24):
        data_df = pd.DataFrame()
        mean_df = pd.DataFrame()
        hours_before_flare = 24 - hour
        # counts = {flare_class: 0 for flare_class in flare_classes}
        subdir = f"0h_{hours_before_flare}h"
        timepoints = hours_before_flare * 5
        for param in FLARE_PROPERTIES:
            df = pd.read_csv(f"{other_directory}/{subdir}/{param}.csv", index_col=0)
            if data_df.empty:
                mean_df["FLARE_TYPE"] = df["FLARE_TYPE"]
                mean_df["COINCIDENCE"] = df["COINCIDENCE"]
                mean_df["LABEL"] = df["FLARE_TYPE"].apply(get_ar_class)
            mean_df[param] = df[get_features([param], high=timepoints+1)].mean(axis=1)

        mean_df.to_csv(f"{other_directory}mean_datasets/nbcmx_{subdir}_mean_timepoint_filtered.csv")


def year_1_report_bcmx_classification_comparison():
    names = [
        "KNN",
        "LR",
        "RFC",
        "SVM",
        # "DART",
        # "LDA"
    ]

    # def get_tss(y_true, y_pred):
    #     cm = confusion_matrix(y_true, y_pred)
    #     tn, fp, fn, tp = cm.ravel()
    #     detection_rate = tp / float(tp + fn)
    #     false_alarm_rate = fp / float(fp + tn)
    #     tss = detection_rate - false_alarm_rate
    #     return tss
    #
    # classifiers = [
    #     KNeighborsClassifier(n_neighbors=3),
    #     LogisticRegression(C=1000, class_weight="balanced"),
    #     RandomForestClassifier(n_estimators=120),
    #     SVC(),
    #     # lgb.LGBMClassifier(boosting_type="dart"),
    #     # LinearDiscriminantAnalysis()
    # ]
    # timepoint_df = pd.read_csv(
    #     f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_24h_default_timepoint_without_filter.csv")
    # mean_df = pd.read_csv(
    #     f"{other_directory}mean_datasets/nbcmx_0h_24h_mean_timepoint.csv")
    # timeseries_df = pd.read_csv(f"{other_directory}0h_24h/r_value.csv")
    # # tss_df = pd.DataFrame(columns=["name", "tss", "timepoint", "dataset"])
    # tss_df = pd.DataFrame(columns=["name", "tss", "dataset"])
    # # tss_df = pd.DataFrame(columns=["name", "tss", "std", "label", "dataset"])
    # for df, label in zip([mean_df, timepoint_df, timeseries_df], ["0h-24h Parameter Mean",
    #                                                               "24h in Advance Timepoint",
    #                                                               "0h-24h Time Series, R_VALUE"]):
    #
    # # timepoint_labels = ["default_timepoint_with_filter", "default_timepoint_without_filter", "nearest_timepoint_with_filter", "nearest_timepoint_without_filter"]
    # # for i in range(0, 25):
    # # for timepoint_label in timepoint_labels:
    # # for i in range(24, 0, -1):
    # #     subdir = f"0h_{i}h"
    #
    #     # timepoint_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_{24}h_{timepoint_label}.csv")
    #     # timepoint_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}timepoints/singh_nbcmx_data_{i}h_timepoint.csv")
    #     # timepoint_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_{i}h_nearest_timepoint_with_filter.csv")
    #     # timepoint_df = pd.read_csv(
    #     #     f"{other_directory}mean_datasets/nbcmx_{subdir}_mean_timepoint.csv")
    #
    #     # time_series_df = pd.DataFrame()
    #     # for param in FLARE_PROPERTIES:
    #     #     if time_series_df.empty:
    #     #         time_series_df = pd.concat([
    #     #             time_series_df,
    #     #             pd.read_csv(f"{other_directory}{subdir}/{param}.csv", index_col=0)],
    #     #             axis=1)
    #     #     else:
    #     #         d = pd.read_csv(f"{other_directory}{subdir}/{param}.csv", index_col=0)
    #     #         d = d.drop(["FLARE_TYPE", "COINCIDENCE"], axis=1)
    #     #         time_series_df = pd.concat([time_series_df, d], axis=1)
    #
    #     all_df = timepoint_df
    #     # all_df = all_df.loc[all_df["xray_class"] != "C"]
    #     coin_df = all_df.loc[all_df["COINCIDENCE"] == True]
    #     noncoin_df = all_df.loc[all_df["COINCIDENCE"] == False]
    #     # for df, label in zip([all_df, coin_df, noncoin_df],
    #     #                      ["All Flares",
    #     #                       "Coincidental Flares",
    #     #                       "Noncoincidental Flares"]):
    #     #     print(label)
    #     if "xray_class" not in df.columns:
    #         xray_class = "FLARE_TYPE"
    #     else:
    #         xray_class = "xray_class"
    #
    #     df = df.loc[df[xray_class] != "N"]
    #     if "LABEL" not in df.columns:
    #         df["LABEL"] = df[xray_class].apply(get_ar_class)
    #
    #     for name, clf in zip(names, classifiers):
    #         tss_list = []
    #         if "R_VALUE" in label:
    #             features = get_features(["R_VALUE"])
    #         else:
    #             features = FLARE_PROPERTIES
    #         X = df[features]
    #         if "R_VALUE" not in label:
    #             X = (X - X.min()) / (X.max() - X.min())
    #         y = df["LABEL"]
    #         for index in range(30):
    #             X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                                 test_size=0.30,
    #                                                                 stratify=df[xray_class])
    #
    #             clf.fit(X_train, y_train)
    #             y_pred = clf.predict(X_test)
    #             tss = get_tss(y_test, y_pred)
    #             tss_list.append(tss)
    #         tss_df.loc[tss_df.shape[0]] = [name, np.mean(tss_list), label]
    #         # tss_df.loc[tss_df.shape[0]] = [name, np.mean(tss_list), np.std(tss_list), timepoint_label, label]
    #         print(tss_df)
    # tss_df.to_csv(f"{other_directory}bcmx_classification_results_year1_report_default_unfilter.csv")
    # # tss_df.to_csv(f"{other_directory}24h_timepoint_method_filter_comparison.csv")



    df = pd.DataFrame(columns=["name", "score", "performance"])
    names = ["KNN", "RFC", "LR", "SVM"]
    # timepoint_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_24h_nearest_timepoint_without_filter.csv")
    # df = pd.read_csv(r"C:\Users\youar\PycharmProjects\flare_forecasting\results\time_series_goodness_of_fit\other\0h_to_24h_timepoint_coincidence_classification.csv")
    # df = pd.read_csv(f"{other_directory}24h_timepoint_method_filter_comparison.csv")
    for filename in ["default_filter", "default_unfilter", "nearest_filter", "nearest_unfilter"]:
        df = pd.read_csv(f"{other_directory}bcmx_classification_results_year1_report_{filename}.csv")
        # df = df.loc[df["label"] == "default_timepoint_without_filter"]
        # df = df.loc[df["timepoint"] == 24]
        # df = tss_df

        df = df.loc[(df["name"] != "DART") & (df["name"] != "LDA")]
        ax = sns.barplot(data=df, x="name", y="tss", hue="dataset")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        # ax.set_title("NBC vs. MX, 24h in Advance Timepoint, Coincidence Comparison", loc="left")
        ax.set_title(f"BC vs. MX, All Flares, Dataset Comparison\n{filename}", loc="left")
        ax.set_ylim(bottom=0.0, top=1.0)
        plt.tight_layout()
        plt.savefig(f"{figure_directory}bcmx_classification_{filename}.png")
        plt.show()
        plt.clf()




def time_point_comparison():
    data_df = pd.read_csv(r"C:\Users\youar\PycharmProjects\flare_forecasting\flare_data\singh_nbcmx_data_corrected.csv", index_col=0)
    data_df["LABEL"] = data_df["xray_class"].apply(get_ar_class)
    # data_df = data_df.loc[data_df["xray_class"] != "N"]
    X = data_df[FLARE_PROPERTIES + ["COINCIDENCE"]]
    y = data_df["LABEL"]

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

    series_tss = {name: [] for name in names}
    for index in range(30):
        for name, clf in zip(names, classifiers):
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size = 0.30, stratify=data_df["xray_class"])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            tss_series = calc_tss(y_test, y_pred)
            series_tss[name].append(tss_series)
    for name in names:
        print(name, np.mean(series_tss[name]), np.std(series_tss[name]))


def individual_flares_plot2():
    m_list.reset_index(inplace=True)
    m_list.drop("index", axis=1, inplace=True)

    print(m_list.shape[0])

    shapes = {f"sum_{hour}": [] for hour in range(0, 24)}
    count = {f"timepoint_{i}": 0 for i in range(1, 121)}
    for index, row in x_list.iterrows():
        nar = row["nar"]
        flare_class = row["xray_class"]
        coincidence = row["COINCIDENCE"]
        time_start = row["time_start"] - timedelta(hours=24)
        time_end = row["time_start"]
        try:
            nar_data = mx_data.loc[data["NOAA_AR"] == nar]
            start_index = nar_data.index.get_indexer([time_start], method='backfill')
            end_index = nar_data.index.get_indexer([time_end],
                                                   method='pad')
            start_index = nar_data.iloc[start_index].index[0]
            end_index = nar_data.iloc[end_index].index[0]
            time_series_df = nar_data.loc[start_index:end_index].reset_index()
            time_series_df = time_series_df.iloc[0:120]
            # print(time_series_df)
            time_series_df.index = [i for i in range(time_series_df.shape[0])]
            # print(time_series_df)

            for hour in range(0, 24):
                shapes[f"sum_{hour}"].append(time_series_df.iloc[hour*5:hour*5 + 5].shape[0])

        except IndexError:
            # print(f"Skipping {flare_class} flare at {time_end} due to IndexError")
            print(time_series_df)
            exit(1)
            continue
        except pd.errors.InvalidIndexError:
            # print(f"Skipping {flare_class} flare at {time_end} due to InvalidIndexError")
            print(time_series_df)
            exit(1)
            continue

    for hour in range(0, 24):
        print(f"{hour},{np.mean(shapes[f'sum_{hour}'])}")


def floor_minute(time, cadence=12):
    return time - timedelta(minutes=time.minute % cadence)

def dropouts():

    for class_list, class_data, c in zip(
            # [m_list, x_list, c_list, b_list, n_list],
            # [mx_data, mx_data, bc_data, b_data, n_data],
            # ["m", "x", "c", "b", "n"]
            [x_list, m_list, c_list, b_list, n_list],
            [mx_data, mx_data, bc_data, b_data, n_data],
            ["x", "m", "c", "b", "n"]
    ):
        class_list.reset_index(inplace=True)
        class_list.drop("index", axis=1, inplace=True)

        dropout_stats = {c: {"isolated_dropout_count": 0,
                             "isolated_pair_dropout_count": 0,
                             "total_flares_with_dropouts": 0,
                             "total_flares_with_isolated_dropouts": 0,
                             "atleast_isolated_pair_dropout_count": 0,
                             "total_dropouts": 0,
                             "distribution": {i: 0 for i in range (0, 122)},
                             "total_dropout_count": [],
                             }
                         for c in ["m", "x", "c", "b", "n"]}

        shapes = {f"sum_{hour}": [] for hour in range(0, 24)}
        for index, row in class_list.iterrows():
            nar = row["nar"]
            time_start = row["time_start"] - timedelta(hours=24)
            time_end = row["time_start"]

            try:
                nar_data = class_data.loc[class_data["NOAA_AR"] == nar]
                start_index = nar_data.index.get_indexer([time_start],
                                                         method='nearest')
                end_index = nar_data.index.get_indexer([time_end],
                                                       method='nearest')
                start_index = nar_data.iloc[start_index].index[0]
                end_index = nar_data.iloc[end_index].index[0]
                time_series_df = nar_data.loc[start_index:end_index]
            except IndexError:
                continue
            except pd.errors.InvalidIndexError:
                continue

            if time_series_df.shape[0] < 120:
                temp_df = pd.DataFrame()
                temp2_df = time_series_df.copy()
                temp_df["timestamp"] = time_series_df.index
                temp_df['minutes'] = temp_df['timestamp'].diff()
                isolated_dropout_count = temp_df.loc[temp_df["minutes"] == pd.Timedelta(minutes=24)].shape[0]
                isolated_pair_dropout_count = temp_df.loc[
                    temp_df["minutes"] == pd.Timedelta(minutes=36)].shape[0]
                temp2_df = temp2_df.reindex(pd.date_range(start=start_index, end=end_index, freq='12T'))
                total_dropouts = int(temp2_df.isna().sum(axis=1).astype(bool).sum())

                if total_dropouts > 0:
                    dropout_stats[c]["total_flares_with_dropouts"] += 1
                if total_dropouts == isolated_dropout_count:
                    dropout_stats[c]["total_flares_with_isolated_dropouts"] += isolated_dropout_count
                if total_dropouts <= isolated_dropout_count + isolated_pair_dropout_count:
                    dropout_stats[c]["atleast_isolated_pair_dropout_count"] += 1
                dropout_stats[c]["total_dropouts"] += total_dropouts
                dropout_stats[c]["isolated_dropout_count"] += isolated_dropout_count
                dropout_stats[c]["isolated_pair_dropout_count"] += isolated_pair_dropout_count

                dropout_df = pd.DataFrame()
                dropout_df.index = pd.date_range(floor_minute(time_start),
                                                 floor_minute(time_end), freq="12min")
                dropout_df["dropout"] = False
                dropouts = pd.date_range(floor_minute(time_start),
                                         floor_minute(time_end), freq="12min").difference(time_series_df.index)

                for dropout in dropouts:
                    dropout_df.loc[dropout]["dropout"] = True
                dropout_df["index"] = [i for i in range(dropout_df.shape[0])]
                dropout_indexes = dropout_df.loc[dropout_df["dropout"] == True]["index"]
                for dropout_index in dropout_indexes:
                    dropout_stats[c]["distribution"][dropout_index] += 1
                dropout_stats[c]["total_dropout_count"].append(len(dropout_indexes))
            else:
                dropout_stats[c]["total_dropout_count"].append(0)


        print(c, dropout_stats[c])
        with open(f"{metrics_directory}{c}_dropout_stats.txt", "w") as fp:
            json.dump(dropout_stats[c], fp, indent=4)

def dropout_statistics():
    for c, class_list in zip(["m", "x", "c", "b", "n", "mx", "bc", "nbc"], [m_list, x_list, c_list, b_list, n_list, mx_list, bc_list, nbc_list]):
        if len(c) > 1:
            with open(f"{metrics_directory}{c[0]}_dropout_stats.txt",
                      "r") as fp:
                d = json.load(fp)
                distribution = d["distribution"]
                dropout_counts = d["total_dropout_count"]
            for char in c[1:]:
                with open(f"{metrics_directory}{char}_dropout_stats.txt",
                          "r") as fp:
                    d = json.load(fp)
                    for i in range(0, 122):
                        distribution[str(i)] += d["distribution"][str(i)]
                    dropout_counts += d["total_dropout_count"]
        else:
            with open(f"{metrics_directory}{c}_dropout_stats.txt", "r") as fp:
                d = json.load(fp)
                distribution = d["distribution"]
                dropout_counts = d["total_dropout_count"]

        distribution.pop('0')
        distribution.pop('120')
        distribution.pop('121')

        # has_dropout = np.count_nonzero(dropout_counts)
        # print(f"{c.upper()}\t{has_dropout}/{class_list.shape[0]}\t{has_dropout / class_list.shape[0]}")

        # print(distribution['6'], c)
        #
        # plt.bar(range(len(distribution)), distribution.values(), align='center')
        # plt.title(f"{c.upper()}-Flare Dropout Distribution")
        # plt.ylabel("Dropout Count")
        # plt.xlabel("Timepoint (from 24h prior to onset)")
        # plt.savefig(f"{figure_directory}dropouts/{c}_dropout_distribution.png")
        # plt.show()

        bins = [1] + [i*5 for i in range(1, 10)]
        arr = plt.hist(dropout_counts, bins=bins)
        plt.title(f"{c.upper()}-Flare Dropout Histogram")
        plt.ylabel("Count")
        plt.xlabel("# of Dropouts over 24h")
        for i in range(len(bins) - 1):
            plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))
        plt.grid(True)
        plt.savefig(f"{figure_directory}dropouts/{c}_dropout_histogram.png")
        plt.show()

def time_series_vector_classification5():
    import lightgbm as lgb
    names = [
        "LR",
        "LDA",
        "KNN",
        "DART",
        "RFC",
        "SVM",
    ]

    classifiers = [
        LogisticRegression(C=1000, class_weight="balanced"),
        LinearDiscriminantAnalysis(),
        KNeighborsClassifier(n_neighbors=3),
        lgb.LGBMClassifier(boosting_type="dart"),
        RandomForestClassifier(n_estimators=120),
        SVC(),
    ]
    coincidences = ["All Flares", "Coincidental Flares",
                    "Noncoincidental Flares"]
    # results_df = pd.DataFrame(columns=["name", "tss_mean", "tss_std", "label"])
    results_df = pd.DataFrame(columns=["name", "tss_mean", "tss_std", "hours", "dataset"])

    # results_df = pd.read_csv(f"{metrics_directory}time_series_classification.csv", index_col=0)
    for hour in range(24, 25):
        subdir = f"0h_{hour}h"
        data_df = pd.DataFrame()
        for param in FLARE_PROPERTIES:
            if data_df.empty:
                data_df = pd.concat([
                    data_df,
                    pd.read_csv(f"{other_directory}{subdir}/{param}.csv", index_col=0)],
                    axis=1)
                # d = data_df.drop(["FLARE_TYPE", "COINCIDENCE"], axis=1)
                # data_df = data_df[["FLARE_TYPE", "COINCIDENCE"]]
                # d.iloc[:, 1:-1:2] = np.NaN  # Single
                # d = d.interpolate(method="linear", axis=1)
                # data_df = pd.concat([data_df, d], axis=1)
            else:
                d = pd.read_csv(f"{other_directory}{subdir}/{param}.csv", index_col=0)
                d = d.drop(["FLARE_TYPE", "COINCIDENCE"], axis=1)
                # d.iloc[:, 1:-1:2] = np.NaN  # Single
                # d = d.interpolate(method="linear", axis=1)
                data_df = pd.concat([data_df, d], axis=1)
        # for coincidence, coin_val in zip(coincidences, [None, True, False]):
        for coincidence, coin_val in zip(["all"], [None]):
            if coin_val is None:
                flare_df = data_df.copy()
            else:
                flare_df = data_df.loc[data_df["COINCIDENCE"] == coin_val]
            flare_df["LABEL"] = flare_df["FLARE_TYPE"].apply(get_ar_class)
            properties_df = flare_df.drop(["FLARE_TYPE", "LABEL", "COINCIDENCE"], axis=1)
            normalized_df = (properties_df - properties_df.min()) / (
                    properties_df.max() - properties_df.min())
            for col in normalized_df.columns:
                flare_df[col] = normalized_df[col]

            series_tss = {name: [] for name in names}
            interpolated_series_tss = {name: [] for name in names}
            df = flare_df

            for clf, name in zip(classifiers, names):
                for index in range(30):
                    train_df, test_df = train_test_split(df, test_size=0.3,
                                                         stratify=flare_df["FLARE_TYPE"])
                    train_X = train_df.drop(["LABEL", "FLARE_TYPE", "COINCIDENCE"], axis=1)
                    test_X = test_df.drop(["LABEL", "FLARE_TYPE", "COINCIDENCE"], axis=1)
                    # print(test_X)
                    interpolated_test_X = test_X.copy()
                    interpolated_test_X.iloc[:, 1:-1:2] = np.NaN  # Single
                    for column in interpolated_test_X.columns:
                        if ("_1" in column and len(column.split('_')[1]) == 1) or "_120" in column:
                            interpolated_test_X[column] = test_X[column]
                    interpolated_test_X = interpolated_test_X.interpolate(method="linear", axis=1)
                    train_y, test_y = train_df["LABEL"].values, test_df["LABEL"].values

                    clf.fit(train_X, train_y)
                    y_pred = clf.predict(test_X)
                    interpolated_y_pred = clf.predict(interpolated_test_X)
                    series_tss[name].append(calc_tss(test_y, y_pred))
                    interpolated_series_tss[name].append(calc_tss(test_y, interpolated_y_pred))
                # results_df.loc[results_df.shape[0]] = [name, np.mean(series_tss[name]), np.std(series_tss[name]), f"{dir[:-1]} time-series"]
                results_df.loc[results_df.shape[0]] = [name, np.mean(series_tss[name]), np.std(series_tss[name]), hour, "Original"]
                results_df.loc[results_df.shape[0]] = [name, np.mean(interpolated_series_tss[name]), np.std(interpolated_series_tss[name]), hour,
                                                       "Interpolated"]
                print(results_df)
                results_df.to_csv(f"{metrics_directory}missing_singles_linear_interpolated_time_series_classification_2.csv")


def timepoint_tss_plot():
    names = [
        "KNN",
        "LR",
        "RFC",
        "SVM",
        # "DART",
        "LDA"
    ]
    coincidences = ["All Flares", "Coincidental Flares", "Noncoincidental Flares"]
    # coincidences = ["All Flares"]
    # df = pd.read_csv(f"{metrics_directory}0h_to_24h_mean_timepoint_coincidence_classification.csv")
    # df = pd.read_csv(f"{metrics_directory}0h_to_24h_timepoint_coincidence_classification.csv")
    # df = pd.read_csv(f"{metrics_directory}0h_to_24h_mean_timepoint_coincidence_classification.csv")
    df = pd.read_csv(f"{other_directory}0h_to_24h_nbcmx_nearest_timepoint_with_filter_coincidence_classification.csv")


    for coincidence in coincidences:
        global_max_tss = -1
        max_index = -1
        for name in names:
            name_df = df.loc[df["name"] == name]
            name_coin_df = name_df.loc[name_df["dataset"] == coincidence]
            tss = list(name_coin_df["tss"])
            # tss = list(name_coin_df["tss_mean"])
            # sd = list(name_coin_df["tss_std"])
            tss.reverse()
            # plt.plot(range(len(tss)), tss, label=name)
            plt.plot(range(len(tss)), tss, label=name)
            plt.ylim(bottom=0.45, top=1)
            plt.xticks(range(0, 25), range(24, -1, -1), rotation="vertical")
            # plt.xticks(range(1, 25), range(24, 0, -1), rotation="vertical")
            # plt.xticks(range(1, 25), rotation="vertical")
            max_tss = max(tss)
            if max_tss > global_max_tss:
                global_max_tss = max_tss
                max_index = tss.index(max_tss)
        plt.axvline(max_index, color="grey", ls="dashed", label="Max TSS", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xlabel("Hours Before Flare Onset")
        plt.ylabel("TSS")
        # plt.title(f"NBC vs. MX, Timepoint Comparison, {coincidence}")
        plt.title(f"NBC vs. MX, Timepoint Comparison, {coincidence}")
        # plt.title(f"NBC vs. MX, Time-series Comparison, {coincidence}")
        # plt.title(coincidence)
        plt.tight_layout()
        plt.savefig(f"{figure_directory}nbcmx_nearest_timepoint_filter_comparison_{coincidence.split()[0].lower()}.png")
        plt.show()
        plt.clf()


def get_dataframe_for_time_series():
    # params = ["R_VALUE", "TOTUSJZ", "TOTUSJH"]
    # df = pd.DataFrame(columns=["FLARE_TYPE", "COINCIDENCE"] + [f"{param}_{i}" for i in range(1, 121) for param in params])
    flare_classes = ["x", "m", "c", "b", "n"]



    for hour in range(0, 1):
        hours_before_flare = 24 - hour
        # counts = {flare_class: 0 for flare_class in flare_classes}
        subdir = f"0h_{hours_before_flare}h"
        timepoints = hours_before_flare * 5
        param_dfs = {
            param: pd.DataFrame(
                columns=["FLARE_TYPE", "COINCIDENCE"] +
                        [f"{param}_{i}" for i in range(1, timepoints + 1)])
            for param in FLARE_PROPERTIES}
        for data, class_list, c in zip(
                [mx_data, mx_data, bc_data, b_data, n_data],
                [x_list, m_list, c_list, b_list, n_list],
                flare_classes
        ):
            print(f'Processing {class_list.shape[0]} {c}-flares for {subdir} ({timepoints} timepoints)')
            # data = data.loc[data["QUALITY"] == 0]
            for index, row in class_list.iterrows():
                nar = row["nar"]
                flare_class = row["xray_class"]
                coincidence = row["COINCIDENCE"]
                time_start = row["time_start"] - timedelta(hours=24)
                time_end = row["time_start"] - timedelta(hours=hour)
                try:
                    nar_data = data.loc[data["NOAA_AR"] == nar]
                    start_index = nar_data.index.get_indexer([time_start],
                                                             method='nearest')
                    end_index = nar_data.index.get_indexer([time_end],
                                                           method='nearest')
                    start_index = nar_data.iloc[start_index].index[0]
                    end_index = nar_data.iloc[end_index].index[0]

                    # time_series_df = nar_data.loc[start_index:end_index].reset_index()

                    time_series_df = nar_data.loc[start_index:end_index]

                    # time_series_df = time_series_df.reindex(
                    #     pd.date_range(start=start_index, end=end_index,
                    #                   freq='12T'))
                    # for param in FLARE_PROPERTIES:
                    #     time_series_df[param] = time_series_df[
                    #         param].interpolate(method="time", limit=2)
                    # # time_series_df = time_series_df.dropna()
                    # if time_series_df.shape[0] < timepoints:
                    #     continue
                    time_series_df = time_series_df.reset_index()
                    time_series_df = time_series_df.iloc[0:timepoints]
                    time_series_df = filter_data(time_series_df, nar)

                    # time_series_df = time_series_df.reindex(
                    #     pd.date_range(start=start_index, end=end_index,
                    #                   freq='12T'))
                    # for param in FLARE_PROPERTIES:
                    #     time_series_df[param] = time_series_df[param].interpolate(
                    #         method="time")
                    time_series_df = time_series_df.dropna()
                    # time_series_df = time_series_df.reset_index()
                    if time_series_df.shape[0] < timepoints:
                        continue

                    time_series_df = time_series_df.iloc[0:timepoints]
                except IndexError:
                    print(
                        f"Skipping {flare_class} flare at {time_end} due to IndexError")
                    continue
                except pd.errors.InvalidIndexError:
                    print(
                        f"Skipping {flare_class} flare at {time_end} due to InvalidIndexError")
                    continue

                for param in FLARE_PROPERTIES:
                    # try:
                    param_values = time_series_df[param].tolist()
                    param_dfs[param].loc[len(param_dfs[param])] = [flare_class,
                                                                   coincidence] + param_values
            #     counts[flare_class.lower()] += 1
            # print(counts)

        for param in FLARE_PROPERTIES:
            param_dfs[param].to_csv(f"{other_directory}/{subdir}/{param.lower()}_no_interpolation.csv")


def get_dataframe_for_mean_time_series():
    # params = ["R_VALUE", "TOTUSJZ", "TOTUSJH"]
    # df = pd.DataFrame(columns=["FLARE_TYPE", "COINCIDENCE"] + [f"{param}_{i}" for i in range(1, 121) for param in params])
    flare_classes = ["x", "m", "c", "b", "n"]
    timestamp_format = "%Y-%m-%d %H:%M:%S"



    for hour in range(1, 24):
        hours_before_flare = 24 - hour
        # counts = {flare_class: 0 for flare_class in flare_classes}
        subdir = f"0h_{hours_before_flare}h"
        timepoints = hours_before_flare * 5
        # param_dfs = {
        #     param: pd.DataFrame(
        #         columns=["FLARE_TYPE", "COINCIDENCE"] +
        #                 [f"{param}_{i}" for i in range(1, hours_before_flare + 1)])
        #     for param in FLARE_PROPERTIES}
        mean_time_series_df = pd.DataFrame(columns=["FLARE_TYPE", "COINCIDENCE"] +
                                           [f"{param}_{i}" for i in range(1, hours_before_flare + 1) for param in FLARE_PROPERTIES])
        flare_count = 0
        param_column_count = mean_time_series_df.shape[1] - 2
        for data, class_list, c in zip(
                [mx_data, mx_data, bc_data, b_data, n_data],
                [x_list, m_list, c_list, b_list, n_list],
                flare_classes
        ):
            print(f'Processing {class_list.shape[0]} {c}-flares for {subdir} ({timepoints} timepoints)')
            # data = data.loc[data["QUALITY"] == 0]
            for index, row in class_list.iterrows():
                nar = row["nar"]
                flare_class = row["xray_class"]
                coincidence = row["COINCIDENCE"]
                mean_time_series_df.loc[flare_count] = [flare_class, coincidence] + [np.nan for _ in range(param_column_count)]
                time_start = floor_minute(row["time_start"] - timedelta(hours=24))
                time_end = floor_minute(row["time_start"] - timedelta(hours=hour))
                temp = filter_data(data, nar)
                nar_data = temp.loc[temp["NOAA_AR"] == nar]
                time_between = nar_data.loc[(nar_data.index >= time_start.strftime(timestamp_format)) & (nar_data.index <= time_end.strftime(timestamp_format))]
                mean_timepoints = []
                for mean_hour in range(hours_before_flare):
                    mean_time_start = time_start + timedelta(hours=mean_hour)
                    mean_time_end = mean_time_start + timedelta(hours=1)
                    timepoints_in_hour = time_between.loc[(time_between.index >= mean_time_start.strftime(timestamp_format)) & (time_between.index < mean_time_end.strftime(timestamp_format))]
                    for param in FLARE_PROPERTIES:
                        mean_timepoints.append(timepoints_in_hour.mean()[param])
                        # mean_timepoints[param] = [timepoints_in_hour.mean()[param]]
                mean_time_series_df.iloc[flare_count, 2:] = mean_timepoints
                flare_count += 1
            mean_time_series_df.to_csv(f"{other_directory}/mean_time_series/{subdir}_mean_time_series_with_null.csv")
            mean_time_series_df.dropna().to_csv(f"{other_directory}/mean_time_series/{subdir}_mean_time_series.csv")

def dropouts2():
    import csv
    empty = True
    def floor_minute(time, cadence=12):
        return time - timedelta(minutes=time.minute % cadence)

    for class_list, class_data, c in zip(
            # [m_list, x_list, c_list, b_list, n_list],
            # [mx_data, mx_data, bc_data, b_data, n_data],
            # ["m", "x", "c", "b", "n"]
            [x_list, m_list, c_list, b_list, n_list],
            [mx_data, mx_data, bc_data, b_data, n_data],
            ["x", "m", "c", "b", "n"]
    ):
        class_list.reset_index(inplace=True)
        class_list.drop("index", axis=1, inplace=True)

        dropout_stats = {c: {
            "missing_singles": 0,
            "missing_pairs": 0,
            "missing_triples": 0,
            "missing_quads": 0,
            "missing_five_plus": 0,
            "flares_only_one_single": 0,
            "flares_only_two_single": 0,
            "flares_only_three_single": 0,
            "flares_only_four_single": 0,
            "flares_five_plus_single": 0,
            "flares_only_one_pair": 0,
            "flares_only_two_pair": 0,
            "flares_only_three_pair": 0,
            "flares_four_plus_pair": 0,
            "flares_only_one_triple": 0,
            "flares_only_two_triple": 0,
            "flares_three_plus_triple": 0,
            "flares_only_one_single_and_one_pair": 0,
            "flares_only_two_single_and_one_pair": 0,
            "flares_only_three_single_and_one_pair": 0,
            "flares_three_plus_single_and_one_pair": 0,
            "flares_only_one_single_and_two_pair": 0,
            "flares_only_two_single_and_two_pair": 0,
            "flares_only_one_single_and_one_triple": 0,
            "flares_only_two_single_and_one_triple": 0,
            "flares_only_one_double_and_one_triple": 0,
        } for c in ["m", "x", "c", "b", "n"]}

        for index, row in class_list.iterrows():
            nar = row["nar"]
            time_start = row["time_start"] - timedelta(hours=24)
            time_end = row["time_start"]

            try:
                nar_data = class_data.loc[class_data["NOAA_AR"] == nar]
                start_index = nar_data.index.get_indexer([time_start],
                                                         method='nearest')
                end_index = nar_data.index.get_indexer([time_end],
                                                       method='nearest')
                start_index = nar_data.iloc[start_index].index[0]
                end_index = nar_data.iloc[end_index].index[0]
                time_series_df = nar_data.loc[start_index:end_index]
            except IndexError:
                continue
            except pd.errors.InvalidIndexError:
                continue

            if time_series_df.shape[0] < 120:
                temp_df = pd.DataFrame()
                temp2_df = time_series_df.copy()
                temp_df["timestamp"] = time_series_df.index
                temp_df['minutes'] = temp_df['timestamp'].diff()

                # Count total dropouts
                isolated_single_dropout_count = temp_df.loc[temp_df["minutes"] == pd.Timedelta(minutes=24)].shape[0]
                isolated_pair_dropout_count = temp_df.loc[temp_df["minutes"] == pd.Timedelta(minutes=36)].shape[0]
                isolated_triple_dropout_count = temp_df.loc[temp_df["minutes"] == pd.Timedelta(minutes=48)].shape[0]
                isolated_quad_dropout_count = temp_df.loc[temp_df["minutes"] == pd.Timedelta(minutes=60)].shape[0]
                isolated_fiveplus_dropout_count = temp_df.loc[temp_df["minutes"] >= pd.Timedelta(minutes=72)].shape[0]

                for dropout_label, dropout_count in zip(
                        ["missing_singles", "missing_pairs", "missing_triples", "missing_quads", "missing_five_plus"],
                        [isolated_single_dropout_count, isolated_pair_dropout_count, isolated_triple_dropout_count, isolated_quad_dropout_count, isolated_fiveplus_dropout_count]):
                    dropout_stats[c][dropout_label] += dropout_count

                # Count flares w/ specific dropouts
                # Singles
                if isolated_single_dropout_count == 1:
                    dropout_stats[c]["flares_only_one_single"] += 1
                elif isolated_single_dropout_count == 2:
                    dropout_stats[c]["flares_only_two_single"] += 1
                elif isolated_single_dropout_count == 3:
                    dropout_stats[c]["flares_only_three_single"] += 1
                elif isolated_single_dropout_count == 4:
                    dropout_stats[c]["flares_only_four_single"] += 1
                elif isolated_single_dropout_count >= 5:
                    dropout_stats[c]["flares_five_plus_single"] += 1

                # Pairs
                if isolated_pair_dropout_count == 1:
                    dropout_stats[c]["flares_only_one_pair"] += 1
                elif isolated_pair_dropout_count == 2:
                    dropout_stats[c]["flares_only_two_pair"] += 1
                elif isolated_pair_dropout_count == 3:
                    dropout_stats[c]["flares_only_three_pair"] += 1
                elif isolated_pair_dropout_count >= 4:
                    dropout_stats[c]["flares_four_plus_pair"] += 1

                # Triples
                if isolated_triple_dropout_count == 1:
                    dropout_stats[c]["flares_only_one_triple"] += 1
                elif isolated_triple_dropout_count == 2:
                    dropout_stats[c]["flares_only_two_triple"] += 1
                elif isolated_triple_dropout_count >= 3:
                    dropout_stats[c]["flares_three_plus_triple"] += 1

                # Combos
                if isolated_single_dropout_count == 1 and isolated_pair_dropout_count == 1:
                    dropout_stats[c]["flares_only_one_single_and_one_pair"] += 1
                elif isolated_single_dropout_count == 2 and isolated_pair_dropout_count == 1:
                    dropout_stats[c]["flares_only_two_single_and_one_pair"] += 1
                elif isolated_single_dropout_count == 3 and isolated_pair_dropout_count == 1:
                    dropout_stats[c]["flares_only_three_single_and_one_pair"] += 1
                elif isolated_single_dropout_count >= 3 and isolated_pair_dropout_count == 1:
                    dropout_stats[c]["flares_three_plus_single_and_one_pair"] += 1
                elif isolated_single_dropout_count == 1 and isolated_pair_dropout_count == 2:
                    dropout_stats[c]["flares_only_one_single_and_two_pair"] += 1
                elif isolated_single_dropout_count == 2 and isolated_pair_dropout_count == 2:
                    dropout_stats[c]["flares_only_two_single_and_two_pair"] += 1
                elif isolated_single_dropout_count == 1 and isolated_triple_dropout_count == 1:
                    dropout_stats[c]["flares_only_one_single_and_one_triple"] += 1
                elif isolated_single_dropout_count == 2 and isolated_triple_dropout_count == 1:
                    dropout_stats[c]["flares_only_two_single_and_one_triple"] += 1
                elif isolated_pair_dropout_count == 1 and isolated_triple_dropout_count == 1:
                    dropout_stats[c]["flares_only_one_double_and_one_triple"] += 1


        print(f"{c.upper()} Flares")
        print(dropout_stats[c].keys())
        print(dropout_stats[c].values())
        print("-" * 50)
        # for k, v in dropout_stats[c].items():
        #     print(k, ":", v)
        # print()
        # print()

        # print(dropout_stats[c])
        # with open(f"{metrics_directory}dropout_stats.csv", "w") as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=dropout_stats[c].keys())
        #     if empty:
        #         writer.writeheader()
        #         empty = False
        #     writer.writerows(dropout_stats[c])
        # with open(f"{metrics_directory}{c}_dropout_stats.txt", "w") as fp:
        #     json.dump(dropout_stats[c], fp, indent=4)


def barplot_counts():
    flare_classes = ['N', 'B', 'C', 'M', 'X']
    timepoints = 25
    index = [f"{i}" for i in range(0, timepoints)]
    all_data = {"N": [],
            "B": [],
            "C": [],
            "M": [],
            "X": []}
    coin_data = {"N": [],
                "B": [],
                "C": [],
                "M": [],
                "X": []}
    noncoin_data = {"N": [],
                "B": [],
                "C": [],
                "M": [],
                "X": []}
    for i in range(0, timepoints):
        temp_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_{i}h_nearest_timepoint_with_filter.csv")
        # if i == 12:
        #     print(temp_df.loc[temp_df["xray_class"] == "X"])
        #     exit(1)
        for flare_class in flare_classes:
            all_data[flare_class].append(temp_df.loc[temp_df["xray_class"] == flare_class].shape[0])
            coin_data[flare_class].append(temp_df.loc[(temp_df["xray_class"] == flare_class) & (temp_df["COINCIDENCE"] == True)].shape[0])
            noncoin_data[flare_class].append(temp_df.loc[(temp_df["xray_class"] == flare_class) & (temp_df["COINCIDENCE"] == False)].shape[0])

    all_df = pd.DataFrame(all_data, index=index)
    coin_df = pd.DataFrame(coin_data, index=index)
    noncoin_df = pd.DataFrame(noncoin_data, index=index)
    print(noncoin_df)
    colors = ["grey", "blue", "green", "orange", "red"]

    # create grouped bar chart
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(24, 18))
    for ax, df, coincidence in zip(axes, [all_df, coin_df, noncoin_df], ["All", "Coincident", "Noncoincident"]):
        df.plot(kind='bar', figsize=(24, 6), width=0.8, stacked=True, color=colors, ax=ax, legend=False)

        # add numerical count labels to each individual bar
        for container in ax.containers:
            ax.bar_label(container, label_type='center', fontsize=10)

        # add labels and title
        ax.set_xlabel('Timepoint Before Flare Onset')
        ax.set_ylabel(coincidence)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    axes[0].set_title('Class Counts by Flare Coincidence\n(Nearest Timepoint w/ Filtering)')
    # add legend
    axes[0].legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

    # show the plot
    fig.savefig(f"{figure_directory}flare_counts_nearest_timepoint_with_filter.png")
    plt.show()


def generate_sinha_timepoint_dataset(all_flare_df: pd.DataFrame=None) -> pd.DataFrame:
    # flare_classes = ["nbc", "mx"]
    flare_classes = ["x", "m", "c", "b", "n"]

    def floor_minute(time, cadence=12):
        if not isinstance(time, str):
            return time - timedelta(minutes=time.minute % cadence)
        else:
            return "not applicable"

    n_list["magnitude"] = 0

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

    for i in range(24, 25):
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

                # temp = filter_data(data, nar)
                temp = data
                ar_records = temp.loc[temp["NOAA_AR"] == nar]
                try:
                    # record_index = ar_records.index[ar_records.index.get_loc(dt, method='nearest')]
                    record_index = ar_records.index[ar_records.index.get_loc(dt)]
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
        all_flare_df.to_csv(f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_{i}h_default_timepoint_without_filter.csv")
        # all_info_df.to_csv(f"{FLARE_DATA_DIRECTORY}singh_nbcmx_data.csv")


def correlation_matrix():
    timepoint_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_24h_default_timepoint_without_filter.csv")
    # timepoint_df = pd.read_csv(f"{other_directory}mean_datasets/nbcmx_0h_12h_mean_timepoint.csv")
    corr = timepoint_df[FLARE_PROPERTIES].corr()
    sns.set(style='white', font_scale=1.5)
    # mask = np.triu(np.ones_like(cm, dtype=bool))
    # mask2 = np.tril(np.ones_like(cm, dtype=bool))
    # binary_cm2 = np.array(timepoint_df[FLARE_PROPERTIES].corr())
    # binary_cm2[binary_cm2 < -0.75] = -1
    # binary_cm2[binary_cm2 > 0.75] = 1
    # binary_cm2[np.logical_or(binary_cm2 != 1, binary_cm2 != -1)] = 0
    plt.figure(figsize=(19, 11), dpi=100)
    cmap = "PiYG"
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, annot=True, fmt='.2f',
                vmin=-1, vmax=1)
    # sns.heatmap(corr, cmap=cmap, square=True, linewidths=.5,
    #             vmin=-1, vmax=1, annot=True, fmt=".2f")
    plt.title("Correlation Matrix, NBCMX, 24h Timepoint, Unfiltered")
    plt.tight_layout()
    plt.savefig(f"{figure_directory}nbcmx_24h_timepoint_correlation_matrix_unfiltered.eps")
    plt.show()

    t = 0.7
    mask = np.abs(corr) >= t
    index = np.where(mask)
    # correlated_params = set()
    # for i, j in zip(index[0], index[1]):
    #     if i != j:
    #         correlated_params.add(corr.index[i])
    #         correlated_params.add(corr.columns[j])
    pairs = set()
    for i, j in zip(index[0], index[1]):
        if i != j and (j, i) not in pairs:
            pairs.add((i, j))
    for i, j in pairs:
        print(f"{corr.index[i]} and {corr.columns[j]} have a correlation of {corr.iloc[i, j]:.2f}")

    # print("Uncorrelated params are:", set(FLARE_PROPERTIES) - correlated_params)
    exit(1)

    # time_series_param_dfs = [pd.read_csv(f"{other_directory}0h_24h/{param}.csv") for param in FLARE_PROPERTIES]
    # for param in FLARE_PROPERTIES:
    #     time_series_param_dfs.append(pd.read)
    # time_series_df = pd.concat([])


def univariate_vs_multivariate():
    timepoint_df = pd.read_csv(
        f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_24h_default_timepoint_with_filter.csv")
    mean_timepoint_df = pd.read_csv(
        f"{other_directory}mean_datasets/nbcmx_0h_24h_mean_timepoint.csv")
    time_series_df = pd.concat(
        [pd.read_csv(f"{other_directory}0h_24h/{param}.csv") for param in FLARE_PROPERTIES],
        axis=1
    )
    print(time_series_df)
    exit(1)
    df = mean_timepoint_df
    if "LABEL" not in df.columns:
        df["LABEL"] = df["xray_class"].apply(get_ar_class)
    xray_class = "xray_class" if "xray_class" in df else "FLARE_TYPE"
    properties_df = df[FLARE_PROPERTIES]
    normalized_df = (properties_df - properties_df.min()) / (
            properties_df.max() - properties_df.min())

    tss_df = pd.DataFrame(columns=["param", "tss", "std"])
    clf = LogisticRegression(C=1000)

    y = df["LABEL"]
    for param in FLARE_PROPERTIES:
        X = np.array(normalized_df[param]).reshape(-1, 1)
        tss_list = []
        for train_index in range(30):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=df[xray_class])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            tss_list.append(calc_tss(y_test, y_pred))
        tss_mean = np.mean(tss_list)
        tss_std = np.std(tss_list)
        tss_df.loc[tss_df.shape[0]] = [param, tss_mean, tss_std]

    tss_df = tss_df.sort_values(by="tss")
    # plt.barh(tss_df["param"], tss_df["tss"], xerr=tss_df["std"], height=0.3, align='center', capsize=3, color='steelblue')
    plt.barh(tss_df["param"], tss_df["tss"], height=0.3,
             align='center', capsize=3, color='steelblue')
    for i, v in enumerate(tss_df["tss"]):
        plt.text(v + 0.1, i, f"{v:.2f}", color='black', fontsize=10, va='center')
    plt.tick_params(axis='y', labelsize=10)
    plt.xlabel("TSS")
    plt.ylabel("AR Parameter")
    plt.title("Univariate LR Classification, NBC vs. MX, 24h Mean Timepoint")
    plt.tight_layout()
    plt.savefig(f"{figure_directory}nbcmx_24h_mean_timepoint_univariate_lr.png")
    plt.show()

    best_params_df = pd.DataFrame(columns=["count", "tss", "std"])
    tss_df = tss_df.sort_values(by="tss", ascending=False)
    for i in range(0, len(FLARE_PROPERTIES)):
        best_params = tss_df["param"][0:i+1].tolist()
        if i == 0:
            X = np.array(normalized_df[best_params]).reshape(-1, 1)
        else:
            X = np.array(normalized_df[best_params])
        tss_list = []
        for train_index in range(30):
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.3,
                                                                stratify=df[xray_class])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            tss_list.append(calc_tss(y_test, y_pred))
        tss_mean = np.mean(tss_list)
        tss_std = np.std(tss_list)
        best_params_df.loc[best_params_df.shape[0]] = [i, tss_mean, tss_std]

    # plt.barh(best_params_df["count"], best_params_df["tss"], xerr=best_params_df["std"], height=0.3,
    #          align='center', capsize=3, color='steelblue')
    plt.barh(best_params_df["count"] + 1, best_params_df["tss"],
            height=0.3,
             align='center', capsize=3, color='steelblue')
    for index, v in enumerate(best_params_df["tss"]):
        plt.text(v + 0.1, index + 1, f"{v:.2f}", color='black', fontsize=10,
                 va='center')
    plt.tick_params(axis='y', labelsize=10)
    plt.xlabel("TSS")
    plt.ylabel("# of Best Parameters")
    plt.yticks(range(1, 21))
    plt.title("Multivariate LR Classification, NBC vs. MX, 24h Mean Timepoint")
    plt.tight_layout()
    plt.savefig(f"{figure_directory}nbcmx_24h_mean_timepoint_multivariate_lr.png")
    plt.show()

def normalization_effects():
    timepoint_df = pd.read_csv(
        f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_24h_default_timepoint_with_filter.csv")
    X = timepoint_df[FLARE_PROPERTIES].values
    y = timepoint_df["xray_class"].apply(get_ar_class).values
    X_norm = (X - X.min()) / (X.max() - X.min())
    loo = LeaveOneOut()

    names = [
        "LR",
        "LDA",
        "KNN",
        "DART",
        "RFC",
        "SVM",
    ]

    classifiers = [
        LogisticRegression(C=1000, class_weight="balanced"),
        LinearDiscriminantAnalysis(),
        KNeighborsClassifier(n_neighbors=3),
        lgb.LGBMClassifier(boosting_type="dart"),
        RandomForestClassifier(n_estimators=120),
        SVC(),
    ]
    for name, clf in zip(names, classifiers):
        print(name)
        y_pred = []
        y_true = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            y_pred.append(clf.predict(X_test))
            y_true.append(y_test)

        print(f"\tTSS for no normalization:", calc_tss(y_true, y_pred))

        y_pred = []
        y_true = []
        for train_index, test_index in loo.split(X_norm):
            X_train, X_test = X_norm[train_index], X_norm[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            y_pred.append(clf.predict(X_test))
            y_true.append(y_test)
        print(f"\tTSS for normalization:", calc_tss(y_true, y_pred))


def maximal_interpolation():
    results_df = pd.DataFrame(
        columns=["name", "tss_mean", "tss_std", "dataset"])
    names = [
        "KNN",
        "RFC",
        "LR",
        "SVM",
        "LDA",
        "DART"
    ]

    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(n_estimators=120),
        LogisticRegression(C=1000),
        SVC(),
        LinearDiscriminantAnalysis(),
        lgb.LGBMClassifier(boosting_type="dart"),
    ]

    flare_df = pd.DataFrame()
    interpolated_df = pd.DataFrame()
    for coincidence, coin_label in zip([False], ["noncoincident"]):
        for param in FLARE_PROPERTIES:
            if flare_df.empty:
                flare_df = pd.read_csv(f"{other_directory}0h_24h/{param}.csv", index_col=0)
                interpolated_df = flare_df.copy()
                for i in range(2, 120, 1):
                    start_col = flare_df[f"{param}_{i - 1}"]
                    end_col = flare_df[f"{param}_{i + 1}"]
                    midpoint = (start_col + end_col) / 2
                    interpolated_df[f"{param}_{i}"] = midpoint
                    # start_col = flare_df[f"{param}_{i}"]
                    # end_col = flare_df[f"{param}_{i+3}"]
                    # sum_col = start_col + end_col
                    # increment = sum_col / 3
                    # interpolated_df[f"{param}_{i+1}"] = flare_df[f"{param}_{i}"] + increment
                    # interpolated_df[f"{param}_{i + 2}"] = flare_df[f"{param}_{i}"] + increment * 2
            else:
                df = pd.read_csv(f"{other_directory}0h_24h/{param}.csv", index_col=0)
                df = df[[f"{param}_{i}" for i in range(1, 121)]]
                flare_df = pd.concat([flare_df, df], axis=1)
                interpolated_df = flare_df.copy()
                for i in range(2, 120, 1):
                    start_col = flare_df[f"{param}_{i-1}"]
                    end_col = flare_df[f"{param}_{i + 1}"]
                    midpoint = (start_col + end_col) / 2
                    interpolated_df[f"{param}_{i}"] = midpoint
                    # start_col = flare_df[f"{param}_{i}"]
                    # end_col = flare_df[f"{param}_{i + 3}"]
                    # sum_col = start_col + end_col
                    # increment = sum_col / 3
                    # interpolated_df[f"{param}_{i + 1}"] = flare_df[f"{param}_{i}"] + increment
                    # interpolated_df[f"{param}_{i + 2}"] = flare_df[f"{param}_{i}"] + increment * 2
        flare_df["LABEL"] = flare_df["FLARE_TYPE"].apply(get_ar_class)
        interpolated_df["LABEL"] = flare_df["LABEL"]
        flare_df = flare_df.loc[flare_df["COINCIDENCE"] == coincidence]
        flare_df.reset_index(inplace=True)
        interpolated_df = interpolated_df.loc[interpolated_df["COINCIDENCE"] == coincidence]
        interpolated_df.reset_index(inplace=True)
        properties_df = flare_df.drop(["FLARE_TYPE", "LABEL", "COINCIDENCE"],
                                      axis=1)
        properties_df2 = interpolated_df.drop(["FLARE_TYPE", "LABEL", "COINCIDENCE"],
                                      axis=1)
        normalized_df = (properties_df - properties_df.min()) / (
                properties_df.max() - properties_df.min())
        normalized_df2 = (properties_df2 - properties_df2.min()) / (
                properties_df2.max() - properties_df2.min())
        for col in normalized_df.columns:
            flare_df[col] = normalized_df[col]
            interpolated_df[col] = normalized_df2[col]
        series_tss = {name: [] for name in names}
        interpolated_series_tss = {name: [] for name in names}
        for name, clf in zip(names, classifiers):
            for trial_index in range(30):
                train_df, test_df = train_test_split(flare_df, test_size=0.3,
                                                     stratify=flare_df["FLARE_TYPE"])
                train_X = train_df.drop(["LABEL", "FLARE_TYPE", "COINCIDENCE"], axis=1)
                test_X = test_df.drop(["LABEL", "FLARE_TYPE", "COINCIDENCE"], axis=1)
                interpolated_X = interpolated_df.iloc[test_X.index].drop(["LABEL", "FLARE_TYPE", "COINCIDENCE"], axis=1)
                train_y, test_y = train_df["LABEL"].values, test_df["LABEL"].values

                clf.fit(train_X, train_y)
                y_pred = clf.predict(test_X)
                interpolated_y_pred = clf.predict(interpolated_X)
                series_tss[name].append(calc_tss(test_y, y_pred))
                interpolated_series_tss[name].append(calc_tss(test_y, interpolated_y_pred))

            results_df.loc[results_df.shape[0]] = [name,
                                                   np.mean(series_tss[name]),
                                                   np.std(series_tss[name]),
                                                "Original"]
            results_df.loc[results_df.shape[0]] = [name, np.mean(
                interpolated_series_tss[name]), np.std(
                interpolated_series_tss[name]), "Interpolated"]
            print(results_df)
            results_df.to_csv(f"{metrics_directory}{coin_label}_maximal_linear_interpolation.csv")


def minimal_interpolation():
    results_df = pd.DataFrame(
        columns=["name", "tss_mean", "tss_std", "timepoint"])
    names = [
        "KNN",
        "RFC",
        "LR",
        "SVM",
        "LDA",
        "DART"
    ]

    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(n_estimators=120),
        LogisticRegression(C=1000),
        SVC(),
        LinearDiscriminantAnalysis(),
        lgb.LGBMClassifier(boosting_type="dart"),
    ]

    flare_df = pd.DataFrame()
    for param in FLARE_PROPERTIES:
        if flare_df.empty:
            flare_df = pd.read_csv(f"{other_directory}0h_24h/{param}.csv", index_col=0)
        else:
            df = pd.read_csv(f"{other_directory}0h_24h/{param}.csv", index_col=0)
            df = df[[f"{param}_{i}" for i in range(1, 121)]]
            flare_df = pd.concat([flare_df, df], axis=1)
    flare_df["LABEL"] = flare_df["FLARE_TYPE"].apply(get_ar_class)
    properties_df = flare_df.drop(["FLARE_TYPE", "LABEL", "COINCIDENCE"],
                                  axis=1)
    normalized_df = (properties_df - properties_df.min()) / (
            properties_df.max() - properties_df.min())
    for col in normalized_df.columns:
        flare_df[col] = normalized_df[col]
    for i in range(2, 120):
        interpolated_df = flare_df.copy()
        for param in FLARE_PROPERTIES:
            interpolated_df[f"{param}_{i}"] = (flare_df[f"{param}_{i-1}"] + flare_df[f"{param}_{i+1}"])/2
        series_tss = {name: [] for name in names}
        interpolated_series_tss = {name: [] for name in names}
        for name, clf in zip(names, classifiers):
            for trial_index in range(30):
                train_df, test_df = train_test_split(flare_df, test_size=0.3,
                                                     stratify=flare_df["FLARE_TYPE"])
                train_X = train_df.drop(["LABEL", "FLARE_TYPE", "COINCIDENCE"], axis=1)
                test_X = test_df.drop(["LABEL", "FLARE_TYPE", "COINCIDENCE"], axis=1)
                interpolated_X = interpolated_df.iloc[test_X.index].drop(["LABEL", "FLARE_TYPE", "COINCIDENCE"], axis=1)
                train_y, test_y = train_df["LABEL"].values, test_df["LABEL"].values

                clf.fit(train_X, train_y)
                interpolated_y_pred = clf.predict(interpolated_X)
                interpolated_series_tss[name].append(calc_tss(test_y, interpolated_y_pred))
                # if i:
                #     y_pred = clf.predict(test_X)
                #     series_tss[name].append(calc_tss(test_y, y_pred))

            # results_df.loc[results_df.shape[0]] = [name,
            #                                        np.mean(series_tss[name]),
            #                                        np.std(series_tss[name]), i,
            #                                     "Original"]
            results_df.loc[results_df.shape[0]] = [name, np.mean(
                interpolated_series_tss[name]), np.std(
                interpolated_series_tss[name]), i]
            print(results_df)
            results_df.to_csv(f"{metrics_directory}minimal_linear_interpolation.csv")

def verify_linear_interpolation():
    import scipy.stats as stats
    for num_cols, quarter in zip([(1, 31), (31, 61), (61, 91), (91, 121)],
                                 ["24h_18h_before", "18h_12h_before", "12h_6h_before", "6h_0h_before"]):
        results_df = pd.DataFrame({param: [0, 0, 0, 0, 0] for param in FLARE_PROPERTIES})
        results_df.index = ["N", "B", "C", "M", "X"]
        min_df = results_df.copy()
        max_df = results_df.copy()
        avg_df = results_df.copy()
        start_col, end_col = num_cols
        for param in FLARE_PROPERTIES:
            # param = "USFLUX"
            flare_df = pd.read_csv(f"{other_directory}0h_24h/{param}.csv", index_col=False)
            for flare_class in ["N", "B", "C", "M", "X"]:
                cols = [f"{param}_{i}" for i in range(start_col, end_col)]
                df = flare_df.loc[flare_df["FLARE_TYPE"] == flare_class]
                original_df = df[cols]
                missing_df = df[cols].copy()
                missing_df.iloc[:, 1:-1:2] = np.NaN  # Single
                # missing_df.iloc[:, 1:-2:3] = np.NaN  # Pair
                # missing_df.iloc[:, 2:-1:3] = np.NaN  # Pair
                interpolated_df = missing_df.interpolate(method="linear", axis=1)
                # print(missing_df.to_string())
                # print(interpolated_df.to_string())
                # exit(1)
                p_values = []
                for index in range(interpolated_df.shape[0]):
                    # Singles
                    missing_pair_df = interpolated_df.iloc[index] #.iloc[1:-1:2]
                    original_pair_df = original_df.iloc[index] #.iloc[1:-1:2]
                    # Pairs
                    # missing_pair_df = pd.concat([interpolated_df.iloc[index].iloc[1:-2:3],
                    #            interpolated_df.iloc[index].iloc[2:-1:3]], axis=0)
                    # original_pair_df = pd.concat(
                    #                     [original_df.iloc[index].iloc[1:-2:3],
                    #                      original_df.iloc[index].iloc[2:-1:3]],
                    #                     axis=0)
                    # print(missing_pair_df)
                    # print(original_pair_df)
                    # exit(1)
                    try:
                        _, p = stats.wilcoxon(missing_pair_df, original_pair_df)
                        p_values.append(p)
                    except ValueError:
                        pass
                    # if p != 1:
                    #     print(p)
                    #     print("EQUAL")
                    #     exit(1)
                    # if np.isnan(p):
                    #     continue

                    # _, p = stats.ttest_ind(interpolated_df.iloc[index].iloc[1:-1:2],
                    #                   original_df.iloc[index].iloc[1:-1:2])
                    # if index == 0:
                    #     print(flare_class, p)
                    # p_values.append(p)
                # stat, p = combine_pvalues(p_values)
                # results_df.at[flare_class, param] = p
                min_df.at[flare_class, param] = min(p_values)
                max_df.at[flare_class, param] = max(p_values)
                avg_df.at[flare_class, param] = np.mean(p_values)
        # results_df.to_csv(f"{metrics_directory}simple_linear_interpolation_missing_singles_two_sample_ttest_fishers_method.csv")
        min_df.to_csv(
            f"{metrics_directory}simple_linear_interpolation_missing_singles_wilcoxen_{quarter}_minimums.csv")
        max_df.to_csv(
            f"{metrics_directory}simple_linear_interpolation_missing_singles_wilcoxen_{quarter}_maximums.csv")
        avg_df.to_csv(
            f"{metrics_directory}simple_linear_interpolation_missing_singles_wilcoxen_{quarter}_averages.csv")


def plot_interpolation():
    names = [
        "KNN",
        "RFC",
        "LR",
        "SVM",
        "LDA",
        "DART"
    ]
    original_scores = {
        "KNN": 0.7296371239714018,
        "RFC": 0.8192856746097873,
        "LR": 0.7114845938375349,
        "SVM": 0.7350005157870513,
        "LDA": 0.4623546869172597,
        "DART": 0.7980503249458425,
    }
    original_std = {
        "KNN": 0.032992172,
        "RFC": 0.02806303,
        "LR": 0.039170799,
        "SVM": 0.032790379,
        "LDA": 0.045119185,
        "DART": 0.027582952,
    }
    df = pd.read_csv(f"{metrics_directory}minimal_linear_interpolation.csv", index_col=0)
    for name in names:
        min_interpolation_df = df.loc[df["name"] == name]
        plt.title(f"Minimal Linear Interpolation, {name}")
        plt.axhline(original_scores[name], ls="-", color="grey", alpha=0.5, label="No Interpolation")
        plt.axhline(original_scores[name] + original_std[name], color="grey", ls="--", alpha=0.5)
        plt.axhline(original_scores[name] - original_std[name], color="grey", ls="--", alpha=0.5)
        plt.plot(min_interpolation_df["timepoint"], min_interpolation_df["tss_mean"], label="With Interpolation")
        # plt.plot(min_interpolation_df["timepoint"], original_scores[name] - min_interpolation_df["tss_std"], ls="dotted")
        # plt.plot(min_interpolation_df["timepoint"], original_scores[name] + min_interpolation_df["tss_std"], ls="dotted")
        plt.legend()
        plt.savefig(f"{figure_directory}minimal_interpolation_{name}.png")
        plt.show()
        # for timepoint in range(2, 120):

    print(df)


def missing_time_series_classification():
    names = [
        "KNN",
        "RFC",
        "LR",
        "SVM",
        "LDA",
        "DART"
    ]
    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(n_estimators=120),
        LogisticRegression(C=1000),
        SVC(),
        LinearDiscriminantAnalysis(),
        lgb.LGBMClassifier(boosting_type="dart"),
    ]
    subdir = "0h_24h"
    results_df = pd.DataFrame(columns=["name", "tss", "missing_timepoint", "std"])

    data_df = pd.DataFrame()
    for param in FLARE_PROPERTIES:
        if data_df.empty:
            data_df = pd.concat([
                data_df,
                pd.read_csv(f"{other_directory}{subdir}/{param}.csv",
                            index_col=0)],
                axis=1)
        else:
            d = pd.read_csv(f"{other_directory}{subdir}/{param}.csv",
                            index_col=0)
            d = d.drop(["FLARE_TYPE", "COINCIDENCE"], axis=1)
            data_df = pd.concat([data_df, d], axis=1)
    # for coincidence, coin_val in zip(["all", "coincident", non], [None, True, False]):
    for missing_timepoint in ["117"]:
        flare_df = data_df.copy()
        flare_df.drop([f"{param}_4" for param in FLARE_PROPERTIES], axis=1, inplace=True)
        flare_df.drop([f"{param}_13" for param in FLARE_PROPERTIES], axis=1,
                      inplace=True)
        flare_df.drop([f"{param}_30" for param in FLARE_PROPERTIES], axis=1,
                      inplace=True)
        flare_df["LABEL"] = flare_df["FLARE_TYPE"].apply(get_ar_class)
        properties_df = flare_df.drop(["FLARE_TYPE", "LABEL", "COINCIDENCE"],
                                      axis=1)
        normalized_df = (properties_df - properties_df.min()) / (
                properties_df.max() - properties_df.min())
        for col in normalized_df.columns:
            flare_df[col] = normalized_df[col]

        series_tss = {name: [] for name in names}
        df = flare_df

        for clf, name in zip(classifiers, names):
            for index in range(30):
                train_df, test_df = train_test_split(df, test_size=0.3,
                                                     stratify=flare_df[
                                                         "FLARE_TYPE"])
                train_X = train_df.drop(["LABEL", "FLARE_TYPE", "COINCIDENCE"],
                                        axis=1)
                test_X = test_df.drop(["LABEL", "FLARE_TYPE", "COINCIDENCE"],
                                      axis=1)
                train_y, test_y = train_df["LABEL"].values, test_df["LABEL"].values
                clf.fit(train_X, train_y)
                y_pred = clf.predict(test_X)
                series_tss[name].append(calc_tss(test_y, y_pred))
            # results_df.loc[results_df.shape[0]] = [name, np.mean(series_tss[name]), np.std(series_tss[name]), f"{dir[:-1]} time-series"]
            results_df.loc[results_df.shape[0]] = [name,
                                                   np.mean(series_tss[name]),
                                                   missing_timepoint,
                                                   np.std(series_tss[name])]
            print(results_df)
            results_df.to_csv(
                f"{metrics_directory}missing_timepoint_classifier_117_2.csv")


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
    # for clf, name in zip(classifiers, names):
    #     clf.fit(train_X, train_y)
    #     y_pred = clf.predict(test_X)
    #     tss_series = calc_tss(test_y, y_pred)
    #     clf.fit(train_X_mean, train_y_mean)
    #     y_pred_mean = clf.predict(test_X_mean)
    #     tss_mean = calc_tss(test_y_mean, y_pred_mean)
    #     print(f"{name} Time-series TSS: {tss_series}")
    #     print(f"{name} Mean Timepoint TSS: {tss_mean}")
    #     print()
    # print()


def correlation_classification():
    timepoint_df = pd.read_csv(
        f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_24h_default_timepoint_with_filter.csv")
    mean_timepoint_df = pd.read_csv(
        f"{other_directory}mean_datasets/nbcmx_0h_24h_mean_timepoint.csv")
    df = mean_timepoint_df
    if "LABEL" not in df.columns:
        df["LABEL"] = df["xray_class"].apply(get_ar_class)
    xray_class = "xray_class" if "xray_class" in df else "FLARE_TYPE"
    properties_df = df[FLARE_PROPERTIES]
    normalized_df = (properties_df - properties_df.min()) / (
            properties_df.max() - properties_df.min())
    clf = LogisticRegression(C=1000)

    y = df["LABEL"]

    correlation_matrix = df.corr()
    mask = correlation_matrix >= 0.7
    columns_to_drop = ["AREA_ACR", "TOTPOT", "TOTUSJZ", "USFLUX", "g_s", "MEANGBZ", "SAVNCPP", "MEANGAM"]
    # columns_to_drop = set()
    # for column in FLARE_PROPERTIES:
    #     correlated_columns = correlation_matrix.index[
    #         mask.loc[:, column]].tolist()
    #     for correlated_column in correlated_columns:
    #         if column != correlated_column:
    #             columns_to_drop.add(correlated_column)
    # df = df.drop(columns_to_drop, axis=1)
    # print(set(FLARE_PROPERTIES) - columns_to_drop)
    X = normalized_df[columns_to_drop]
    print(X.columns)
    tss_list = []
    for train_index in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3,
                                                            stratify=df[xray_class])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        tss_list.append(calc_tss(y_test, y_pred))
    tss_mean = np.mean(tss_list)
    tss_std = np.std(tss_list)

    print(f"Mean: {tss_mean}, Std: {tss_std}")


singh_filename = r"C:\Users\youar\PycharmProjects\flare_forecasting\flare_data\singh_nabmx_24h_default_timepoint_without_filter.csv"
def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    warnings.simplefilter("ignore", category=SettingWithCopyWarning)
    get_dataframe_of_vectors()
    # missing_time_series_classification()
    # get_dataframe_of_vectors()
    # correlation_matrix()
    # correlation_classification()
    exit(1)
    # plot_interpolation()
    # generate_parallel_coordinates()
    # exit(1)
    # maximal_interpolation()
    # get_idealized_flare()
    # idealized_flares_plot()
    # exit(1)
    # print(mx_data)
    # get_idealized_flare()
    # individual_flares_plot()
    # idealized_flares_plot()
    # exit(1)
    # barplot_counts()
    # year_1_report_bcmx_classification_comparison()
    # timepoint_tss_plot()
    # classification_plot()
    # time_series_vector_classification5()



    flare_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}singh_nbcmx_data_corrected.csv", index_col=False)
    # flare_df = pd.read_csv(singh_filename, index_col=0, parse_dates=["time_start"])

    # flare_df = pd.read_csv("apj_singh_dataset.csv", index_col=0,  parse_dates=["time_start"])
    # flare_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}timepoints_default/singh_nbcmx_data_24h_nearest_timepoint_with_filter.csv", index_col=0,  parse_dates=["time_start"])


    # flare_df = flare_df.loc[flare_df["xray_class"] != "A"]
    for coincidence in ["all", "coincident", "noncoincident"]:
        if coincidence == "coincident":
            df = flare_df.loc[flare_df["COINCIDENCE"] == True]
        elif coincidence == "noncoincident":
            df = flare_df.loc[flare_df["COINCIDENCE"] == False]
        else:
            df = flare_df
        a = df.loc[(df["xray_class"] == "A")].shape[0]
        b = df.loc[(df["xray_class"] == "B")].shape[0]
        c = df.loc[(df["xray_class"] == "C")].shape[0]
        m = df.loc[(df["xray_class"] == "M")].shape[0]
        x = df.loc[(df["xray_class"] == "X")].shape[0]
        n = df.loc[(df["xray_class"] == "N")].shape[0]

        print(coincidence, n, a, b, m, x, a + n + b + m + x)
    print()
    # generate_parallel_coordinates("noncoincident", flare_df)
    for coincidence, val in zip(["coincident", "noncoincident"], [True, False]):
        flare_df = flare_df.loc[flare_df["xray_class"] != "A"]
        # df = flare_df.loc[flare_df["COINCIDENCE"] == val]
        # df.mean().to_csv(f"{metrics_directory}{coincidence}_mean_parameters.csv")
        # print(flare_df.mean())
        generate_parallel_coordinates(coincidence, flare_df)
    exit(1)
    # goodness_of_fit2()
    # time_point_comparison()
    # generate_sinha_timepoint_dataset()
    # barplot_counts()
    # year_1_report_bcmx_classification_comparison()
    # dropouts()
    # get_dataframe_for_time_series()
    # generate_mean_dataset()
    # year_1_report_bcmx_classification_comparison()
    # year_1_report_bcmx_classification_comparison()
    # year_1_report_bcmx_classification_comparison()
    # time_series_vector_classification5()
    # time_series_vector_classification5("interpolated_time_series/0h_24h/")
    # time_series_vector_classification5()
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
