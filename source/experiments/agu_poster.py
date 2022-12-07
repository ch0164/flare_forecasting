import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from source.common_imports import *
from source.constants import *
from source.utilities import *


coincidences = ["all", "coincident", "noncoincident"]
flare_classes = ["B", "C", "M", "X"]
colors = ["cyan", "lime", "orange", "red"]

study_caption = "Solar Cycle 24, Peak Years (2013-2014)"
# agu_properties = [
#     'ABSNJZH',
#     'AREA_ACR',
#     'MEANGAM',
#     'MEANGBH',
#     'MEANGBT',
#     'MEANGBZ',
#     'MEANJZD',
#     'MEANJZH',
#     'MEANPOT',
#     'MEANSHR',
#     'R_VALUE',
#     'SAVNCPP',
#     'SHRGT45',
#     'TOTPOT',
#     'TOTUSJH',
#     'TOTUSJZ',
#     'USFLUX',
# ]
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
agu_properties = SINHA_PARAMETERS

# Experiment Name (No Acronyms)
experiment = "agu_poster"
experiment_caption = experiment.title().replace("_", " ")

# Place any results in the directory for the current experiment.
now_string, figure_directory, metrics_directory, other_directory = \
    build_experiment_directories(experiment)


def generate_flare_count_line_plot(coincidence, all_flares_df):
    fig, ax = plt.subplots()
    if coincidence == "coincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
    elif coincidence == "noncoincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]
    else:
        flares_df = all_flares_df
    s = 0
    plot_offsets_x = np.array([-2, -1, 1, 2]) * 1
    for label, color, offset in zip(flare_classes, colors, plot_offsets_x):
        flare_df = flares_df.loc[flares_df["xray_class"] == label]
        values, counts = np.unique(flare_df["nar"], return_counts=True)
        # print(label, sum(counts))
        value_counts = [(value, count) for value, count in
                        zip(values, counts)]
        value_counts = sorted(value_counts,
                              key=lambda value_count: value_count[0],
                              reverse=True)
        values = [value + offset for value, _ in value_counts]
        counts = [count for _, count in value_counts]
        ax.bar(values, counts, color=color, label=label)
        s += len(values)

    ax.set_title(f"{coincidence.capitalize()}")
    ax.set_xlabel("AR #")
    ax.set_ylabel("# of Flares")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=11350, right=12500)
    ax.set_ylim(bottom=0, top=33)
    plt.tight_layout()
    plt.savefig(f"{figure_directory}{coincidence}_bcmx_flare_count_ar_plot.png")
    plt.show()


def generate_parallel_coordinates(coincidence, all_flares_df):
    fig, ax = plt.subplots(figsize=(19, 10))
    if coincidence == "coincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
    elif coincidence == "noncoincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]
    else:
        flares_df = all_flares_df
    properties_df = flares_df[agu_properties]
    # normalized_df = (properties_df - properties_df.mean()) / properties_df.std()
    normalized_df = (properties_df - properties_df.min()) / (
                properties_df.max() - properties_df.min())
    normalized_df["xray_class"] = flares_df["xray_class"]

    print()
    normalized_df.sort_values(by="xray_class")
    parallel_coordinates(normalized_df, "xray_class", agu_properties, ax, sort_labels=flare_classes,
                         color=colors, axvlines=True, axvlines_kwds={"color": "white"},
                         alpha=0.7)
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


def generate_time_plot(coincidence, all_flares_df):
    years = [2012, 2013, 2014, 2015, 2016]
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    new_index = [f"{quarter} {year}" for year in years for quarter in quarters]
    if coincidence == "coincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
    elif coincidence == "noncoincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]
    else:
        flares_df = all_flares_df

    # plt.figure(figsize=(25, 25))
    flare_df = pd.DataFrame(columns=["B", "C", "M", "X"])
    for year in years:
        for month in range(1, 12 + 1, 3):
            flares = flares_df.loc[(flares_df['time_start'].dt.year == year)]
            flares = pd.concat([flares.loc[flares["time_start"].dt.month == month],
                               flares.loc[flares["time_start"].dt.month == month + 1],
                               flares.loc[flares["time_start"].dt.month == month + 2]])

            print(flares.loc[(flares["xray_class"] == "M") |
                             (flares["xray_class"] == "X")])
            # exit(1)

            b = flares.loc[(flares["xray_class"] == "B")].shape[0]
            c = flares.loc[(flares["xray_class"] == "C")].shape[0]
            m = flares.loc[(flares["xray_class"] == "M")].shape[0]
            x = flares.loc[(flares["xray_class"] == "X")].shape[0]
            flare_counts = [b, c, m, x]
            flare_df.loc[len(flare_df)] = flare_counts

    # coin_flare_df.index = new_index
    flare_df.index = new_index
    # coin_flare_df.plot(kind="bar", stacked=True, color=colors)
    flare_df.plot(kind="bar", stacked=True, color=colors)
    plt.title(f"{coincidence.capitalize()}")
    plt.xticks(rotation="vertical", ha="center")
    plt.ylim(bottom=0, top=100)
    plt.ylabel("# of Flares")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{figure_directory}{coincidence}_bcmx_flare_count_month_plot.png")
    plt.show()


def generate_statistics_tables(all_flares_df):
    for coincidence in coincidences:
        flares_df = all_flares_df.copy()
        if coincidence == "coincident":
            flares_df = flares_df.loc[flares_df["COINCIDENCE"] == True]
        elif coincidence == "noncoincident":
            flares_df = flares_df.loc[flares_df["COINCIDENCE"] == False]
        # flares_df.drop("level_0", inplace=True, axis=1)

        b = flares_df.loc[(flares_df["xray_class"] == "B")][agu_properties]
        m = flares_df.loc[(flares_df["xray_class"] == "M")][agu_properties]
        x = flares_df.loc[(flares_df["xray_class"] == "X")][agu_properties]
        flare_dfs = [b, m, x]
        all_classes_df = pd.DataFrame(columns=agu_properties)
        for flare_df in [b, m, x]:
            means = flare_df.mean().values
            all_classes_df.loc[len(all_classes_df)] = means
        all_classes_df.index = flare_classes
        all_classes_df = all_classes_df.T
        all_classes_df.index.names = ['parameter']
        all_classes_df.to_csv(f"{other_directory}{coincidence}_bcmx_class_means.csv")



def simple_classification(all_flares_df, include_c=False):
    fig, ax = plt.subplots(ncols=3, figsize=(8, 4))
    for axis_index, coincidence in enumerate(coincidences):
        if coincidence == "coincident":
            flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
        elif coincidence == "noncoincident":
            flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]
        else:
            flares_df = all_flares_df

        b = flares_df.loc[(flares_df["xray_class"] == "B")]
        c = flares_df.loc[(flares_df["xray_class"] == "C")]
        m = flares_df.loc[(flares_df["xray_class"] == "M")]
        x = flares_df.loc[(flares_df["xray_class"] == "X")]

        means = pd.DataFrame(columns=agu_properties)
        stds = pd.DataFrame(columns=agu_properties)
        if include_c:
            flares = [b, c, m, x]
            labels = ["B", "C", "M", "X"]
            classifications = ["BC", "MX"]
        else:
            flares = [b, m, x]
            labels = ["B", "M", "X"]
            classifications = ["B", "MX"]
        for df in flares:
            means.loc[len(means)] = df.mean()
            stds.loc[len(stds)] = df.std()
        means.index = labels
        stds.index = labels

        def predict(x, param):
            param_means = means[param]
            param_stds = stds[param]
            # if param_means.loc["B"] - param_stds.loc["B"] <= x <= \
            #         param_means.loc["B"] + param_stds.loc["B"]:
            #     return "B"
            # if param_means.loc["M"] - param_stds.loc["M"] <= x <= \
            #         param_means.loc["M"] + param_stds.loc["M"]:
            #     return "M"
            # else:
            #     return "X"
            if x <= param_means.loc["M"] - param_stds.loc["M"]:
                if include_c:
                    return "BC"
                else:
                    return "B"
            else:
                return "MX"

        predictions_df = pd.DataFrame(columns=agu_properties)
        for index, row in flares_df.iterrows():
            predictions = []
            for param in agu_properties:
                value = row[param]
                predictions.append(predict(value, param))
            predictions_df.loc[len(predictions_df)] = predictions

        predictions = []
        for i in range(predictions_df.T.shape[1]):
            predictions.append(predictions_df.T[i].mode()[0])
        df = pd.DataFrame({"pred": predictions, "true": flares_df["xray_class"]})

        df.replace("M", "MX", inplace=True)
        df.replace("X", "MX", inplace=True)

        if include_c:
            df.replace("B", "BC", inplace=True)
            df.replace("C", "BC", inplace=True)

        cm, cm_label = confusion_matrix(list(df["true"]), list(df["pred"])), coincidence
        write_classification_metrics(list(df["true"]), list(df["pred"]), "Mean-based Classifier",
                                     classifications, print_output=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=classifications)
        disp.plot(ax=ax[axis_index], colorbar=False)
        tn, fp, fn, tp = cm.ravel()
        detection_rate = tp / float(tp + fn)
        false_alarm_rate = fp / float(fp + tn)
        tss = detection_rate - false_alarm_rate
        disp.ax_.set_title(f"{cm_label.capitalize()} (TSS: {tss:.2f})")
        disp.ax_.set_xlabel("")
        if axis_index != 0:
            disp.ax_.set_ylabel("")
    # disp.im_.colorbar.remove()

    fig.text(0.5, 0.1, "Predicted label", ha='center')
    plt.suptitle("Mean-based Classifier, Confusion Matrices")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.tight_layout()
    plt.show()



    # write_classification_metrics(list(df["true"]), list(df["pred"]),
    #                              f"{metrics_directory}bcmx_{coincidence}_classification_metrics.txt",
    #                              clf_name=f"Mean +/- Std Thresholding on {', '.join(params)}",
    #                              flare_classes=["BC", "MX"])

def lda_classification(all_flares_df, include_c: False):
    fig, ax = plt.subplots(ncols=3, figsize=(8, 4))
    for axis_index, coincidence in enumerate(coincidences):
        if coincidence == "coincident":
            flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
        elif coincidence == "noncoincident":
            flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]
        else:
            flares_df = all_flares_df.copy()

        # flares_df.fillna(0, inplace=True)
        flares_df.dropna(inplace=True)
        flares_df["xray_class"].replace("M", "MX", inplace=True)
        flares_df["xray_class"].replace("X", "MX", inplace=True)
        if include_c:
            classifications = ["BC", "MX"]
            flares_df["xray_class"].replace("C", "BC", inplace=True)
            flares_df["xray_class"].replace("B", "BC", inplace=True)
        else:
            classifications = ["B", "MX"]
            flares_df = flares_df.loc[flares_df["xray_class"] != "C"]

        y = flares_df["xray_class"].to_numpy()
        X = flares_df[FLARE_PROPERTIES]

        loo = LeaveOneOut()
        loo.get_n_splits(X)
        y_true, y_pred = [], []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            train_lda = LinearDiscriminantAnalysis()
            train_components = train_lda.fit_transform(
                X_train, y_train)
            train_lda_df = pd.DataFrame(train_components, columns=["LD1"])
            train_lda_df["xray_class"] = pd.Series(y_train)

            mx_mid = train_lda_df.loc[train_lda_df["xray_class"] == "MX"].mean().values[0]
            b_mid = train_lda_df.loc[train_lda_df["xray_class"] == "B"].mean().values[0]
            test_lda = np.dot(X_test.values.tolist()[0], train_lda.coef_.tolist()[0])
            # print(b_mid, mx_mid)
            # print(test_lda)
            # exit(1)
            if test_lda <= (b_mid + mx_mid) / 2:
                y_pred.append("B")
            else:
                y_pred.append("MX")
            y_true.append(y_test[0])

            # pred = train_lda.predict(X_test[FLARE_PROPERTIES])[0]
            # y_pred.append(pred)


        # lda = LinearDiscriminantAnalysis()
        # for train_index, test_index in loo.split(X):
        #     X_train, X_test = X[train_index], X[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
        #     lda.fit(X_train, y_train)
        #     y_pred.append(lda.predict(X_test)[0])
        #     y_true.append(y_test[0])
        cm = confusion_matrix(y_true, y_pred)
        cm_label = coincidence
        write_classification_metrics(y_true, y_pred,
                                     "LDA-based Classifier",
                                     classifications, print_output=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=classifications)
        disp.plot(ax=ax[axis_index], colorbar=False)
        tn, fp, fn, tp = cm.ravel()
        detection_rate = tp / float(tp + fn)
        false_alarm_rate = fp / float(fp + tn)
        tss = detection_rate - false_alarm_rate
        disp.ax_.set_title(f"{cm_label.capitalize()} (TSS: {tss:.2f})")
        disp.ax_.set_xlabel("")
        if axis_index != 0:
            disp.ax_.set_ylabel("")
    # disp.im_.colorbar.remove()

    fig.text(0.5, 0.1, "Predicted label", ha='center')
    plt.suptitle("LDA-based Classifier, Confusion Matrices")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.tight_layout()
    plt.show()


def best_time_window_plot():
    timepoint = 0.8307320217141415
    mean_24 = 0.8597385827601656
    mean_12 = [
        0.8637692392339232,
        0.8673653011538055,
        0.8702056791944432,
        0.8631563706563707,
        0.8677731854740907,
        0.8622530481056997,
        0.8608562362564555,
        0.8711446360153257,
        0.8696405120286321,
        0.8635299579708728,
        0.8654336109750123,
        0.8566498051649442,
        0.8528160873579598,
    ]
    mean_8 = [
        0.8720472440944882,
        0.8762278978388998,
        0.8683693516699411,
        0.8703339882121808,
        0.8745098039215686,
        0.8722986247544204,
        0.8762278978388998,
        0.8767123287671232,
        0.8740157480314961,
        0.8717948717948718,
        0.8725490196078431,
        0.8750000000000000,
        0.8693957115009746,
        0.8669275929549902,
        0.8590998043052838,
        0.8518518518518519,
        0.8518518518518519,
    ]
    mean_8.reverse()
    mean_12.reverse()
    y = [timepoint] + [mean_24] + mean_12 + mean_8
    x = range(1, len(y) + 1)
    print(y)
    min_y = min(y)
    min_x = y.index(min_y) + 1
    max_y = max(y)
    max_x = y.index(max_y) + 1
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.plot(x, y, "yo-", label="MX Recall")
    plt.plot(min_x, min_y, "ro", label="min")
    plt.plot(max_x, max_y, "go", label="max")
    print(min_x, min_y)
    print(max_x, max_y)
    plt.text(0.1, 0.83, "test")
    plt.axvline(1.5)
    plt.axvline(2.5)
    plt.axvline(14.5)
    plt.xscale("log")
    plt.text(0.075, 0.1, '24hr time point', horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes)
    plt.text(0.22, 0.1, '24hr mean', horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
    plt.text(0.525, 0.1, '12hr means', horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
    plt.text(0.875, 0.1, '8hr means', horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes)
    plt.xticks([])
    plt.ylabel("MX Recall")
    plt.title("B/MX Time Window Comparison, All Flares, 2010-2021")
    plt.legend(loc="upper left")
    fig.show()


def main():
    plt.style.use('dark_background')
    # best_time_window_plot()
    # exit(1)
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    flare_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}agu_bcmx.csv")  # use for bcmx
    # flare_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_nbmx_data.csv")  # use for bmx
    flare_df = flare_df.loc[flare_df["xray_class"] != "N"]
    flare_df = flare_df.loc[flare_df["xray_class"] != "A"]
    # flares_df = flares_df[agu_properties + ["time_start", "xray_class", "COINCIDENCE"]].dropna()

    flares_df = flare_df.sort_values(by="xray_class")
    flares_df = flares_df.loc[
        (flares_df["time_start"].str.contains("2012")) |
                              (flares_df["time_start"].str.contains("2013"))
                              | (flares_df["time_start"].str.contains("2014"))
                              | (flares_df["time_start"].str.contains("2015"))
                              | (flares_df["time_start"].str.contains("2016"))
        ].dropna()
    flares_df["time_start"] = pd.to_datetime(flares_df["time_start"])

    # Plot stuff
    # generate_statistics_tables(flares_df)
    print(flares_df.shape[0])
    # simple_classification(flares_df.copy())
    # lda_classification(flares_df.copy())
    for coincidence in coincidences:
        if coincidence == "coincident":
            df = flares_df.loc[flares_df["COINCIDENCE"] == True]
        elif coincidence == "noncoincident":
            df = flares_df.loc[flares_df["COINCIDENCE"] == False]
        else:
            df = flares_df
        b = df.loc[(df["xray_class"] == "B")].shape[0]
        c = df.loc[(df["xray_class"] == "C")].shape[0]
        m = df.loc[(df["xray_class"] == "M")].shape[0]
        x = df.loc[(df["xray_class"] == "X")].shape[0]

        print(coincidence, b, m, x, b + m + x)
    # simple_classification(flares_df.copy(), True)
    # lda_classification(flares_df.copy(), False)
        # generate_parallel_coordinates(coincidence, flares_df.copy())
        # generate_flare_count_line_plot(coincidence, flares_df.copy())
        generate_time_plot(coincidence, flares_df.copy())


if __name__ == "__main__":
    main()