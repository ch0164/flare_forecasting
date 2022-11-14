import pandas as pd

from source.common_imports import *
from source.constants import *
from source.utilities import *


coincidences = ["all", "coincident", "noncoincident"]
flare_classes = ["B", "C", "M", "X"]
colors = ["blue", "green", "orange", "red"]

study_caption = "Solar Cycle 24, Peak Years (2013-2014)"

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
    for label, color in zip(flare_classes, colors):
        flare_df = flares_df.loc[flares_df["xray_class"] == label]

        values, counts = np.unique(flare_df["nar"], return_counts=True)
        value_counts = [(int(value), count) for value, count in
                        zip(values, counts)
                        if not pd.isna(value)]
        value_counts = sorted(value_counts,
                              key=lambda value_count: value_count[1],
                              reverse=True)
        values = [value for value, _ in value_counts]
        counts = [count for _, count in value_counts]
        values.sort()
        ax.plot(values, counts, color=color, label=label)
    ax.set_title(f"BCMX Flare Count, {coincidence.capitalize()} Flares,\n{study_caption}")
    ax.set_xlabel("AR #")
    ax.set_ylabel("# of Flares")

    plt.tight_layout()
    plt.savefig(f"{figure_directory}{coincidence}_bcmx_flare_count_line_plot.png")
    plt.show()


def generate_time_plot(coincidence, all_flares_df):
    new_index = [
        'Jan 2013',
        'Feb 2013',
        'Mar 2013',
        'Apr 2013',
        'May 2013',
        'Jun 2013',
        'Jul 2013',
        'Aug 2013',
        'Sep 2013',
        'Oct 2013',
        'Nov 2013',
        'Dec 2013',
        'Jan 2014',
        'Feb 2014',
        'Mar 2014',
        'Apr 2014',
        'May 2014',
        'Jun 2014',
        'Jul 2014',
        'Aug 2014',
        'Sep 2014',
        'Oct 2014',
        'Nov 2014',
        'Dec 2014'
    ]
    fig, ax = plt.subplots()
    if coincidence == "coincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
    elif coincidence == "noncoincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]
    else:
        flares_df = all_flares_df

    # plt.figure(figsize=(25, 25))
    flare_df = pd.DataFrame(columns=flare_classes)
    for year in [2013, 2014]:
        for month in range(1, 12 + 1):
            flares = flares_df.loc[
                (flares_df['time_start'].dt.year == year) &
                (flares_df['time_start'].dt.month == month)]

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
    plt.title(f"BCMX Flare Count, {coincidence.capitalize()} Flares,\n{study_caption}")
    plt.xticks(rotation="vertical", ha="center")
    plt.ylabel("# of Flares")
    plt.tight_layout()
    plt.savefig(f"{figure_directory}{coincidence}_bcmx_flare_count_bar_plot.png")
    plt.show()


def simple_classification(coincidence, all_flares_df):
    if coincidence == "coincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
    elif coincidence == "noncoincident":
        flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]
    else:
        flares_df = all_flares_df

    params = ["TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "R_VALUE"]
    b = flares_df.loc[(flares_df["xray_class"] == "B")]
    c = flares_df.loc[(flares_df["xray_class"] == "C")]
    m = flares_df.loc[(flares_df["xray_class"] == "M")]
    x = flares_df.loc[(flares_df["xray_class"] == "X")]
    means = pd.DataFrame(columns=params)
    stds = pd.DataFrame(columns=params)
    for df in [b, c, m, x]:
        means.loc[len(means)] = df.mean()
        stds.loc[len(stds)] = df.std()
    means.index = flare_classes
    stds.index = flare_classes

    def predict(x, param):
        param_means = means[param]
        param_stds = stds[param]
        if param_means.loc["C"] - param_stds.loc["C"] <= x <= \
                param_means.loc["C"] + param_stds.loc["C"]:
            return "C"
        elif param_means.loc["M"] - param_stds.loc["M"] <= x <= \
                param_means.loc["M"] + param_stds.loc["M"]:
            return "M"
        elif param_means.loc["X"] - param_stds.loc["X"] <= x:
            return "X"
        else:
            return "B"

    predictions_df = pd.DataFrame(columns=params)
    for index, row in flares_df.iterrows():
        predictions = []
        for param in params:
            value = row[param]
            predictions.append(predict(value, param))
        predictions_df.loc[len(predictions_df)] = predictions

    predictions = []
    for i in range(predictions_df.T.shape[1]):
        predictions.append(predictions_df.T[i].mode()[0])
    df = pd.DataFrame({"pred": predictions, "true": flares_df["xray_class"]})

    df.replace("B", "BC", inplace=True)
    df.replace("C", "BC", inplace=True)
    df.replace("M", "MX", inplace=True)
    df.replace("X", "MX", inplace=True)

    write_classification_metrics(list(df["true"]), list(df["pred"]),
                                 f"{metrics_directory}bcmx_{coincidence}_classification_metrics.txt",
                                 clf_name=f"Mean +/- Std Thresholding on {', '.join(params)}",
                                 flare_classes=["BC", "MX"])



def main():
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    lo_time = 0
    hi_time = 24
    flare_dataframes = [
        get_ar_properties(flare_class,
                          lo_time,
                          hi_time,
                          filter_multiple_ars=False,
                          )
        for flare_class in ["B", "C", "MX"]
    ]
    b_df, c_df, mx_df = tuple(flare_dataframes)
    m_df = mx_df.loc[mx_df["xray_class"] == "M"]
    x_df = mx_df.loc[mx_df["xray_class"] == "X"]
    c_df = c_df.loc[c_df["xray_class"] == "C"]
    flare_dataframes = [b_df, c_df, m_df, x_df]
    flares_df = pd.concat(flare_dataframes).dropna()
    flares_df.reset_index(inplace=True)
    flares_df.drop("index", axis=1, inplace=True)

    flares_df = flares_df.loc[(flares_df["time_start"].str.contains("2013")) |
                              flares_df["time_start"].str.contains("2014")]
    flares_df["time_start"] = pd.to_datetime(flares_df["time_start"])

    # Plot stuff
    for coincidence in coincidences:
        simple_classification(coincidence, flares_df.copy())
        # generate_flare_count_line_plot(coincidence, flares_df.copy())
        # generate_time_plot(coincidence, flares_df.copy())


if __name__ == "__main__":
    main()