import pandas as pd
from matplotlib import pyplot as plt


from source.constants import RESULTS_DIRECTORY, COINCIDENCES


metrics = ["tss", "recall"]
time_intervals = [1, 2, 4, 8, 12, 24]


if __name__ == "__main__":
    dir = RESULTS_DIRECTORY + "time_window_classification/metrics/"
    plt.subplots(figsize=(20, 13))
    tss_df = pd.DataFrame()
    recall_df = pd.DataFrame()
    shapes = []
    averages = {coin: {"tss": [], "recall": []} for coin in COINCIDENCES}
    for time_interval in time_intervals:
        tss_file = f"{dir}{time_interval}h_nb_mx_lda_tss.csv"
        tss_time_df = pd.read_csv(tss_file).iloc[::-1].drop("Unnamed: 0", axis=1)
        tss_df = pd.concat([tss_df, tss_time_df])
        recall_file = f"{dir}{time_interval}h_nb_mx_lda_recall.csv"
        recall_time_df = pd.read_csv(recall_file).iloc[::-1].drop("Unnamed: 0", axis=1)
        recall_df = pd.concat([recall_df, recall_time_df])
        shapes.append(recall_df.shape[0])

    colors = ["grey", "blue", "red"]

    def get_title(metric):
        return f"Time Window Mean Analysis, {metric} from LDA Classifier Using LOO CV, \n" \
               f"for NB/MX Flares"

    tss_df.reset_index(inplace=True)
    tss_df.drop("index", axis=1, inplace=True)
    recall_df.reset_index(inplace=True)
    recall_df.drop("index", axis=1, inplace=True)

    # Get stats for later.
    recall_mins = list(recall_df.min())
    recall_top_quantiles = list(recall_df.quantile(0.75))
    tss_mins = list(tss_df.min())
    tss_top_quantiles = list(tss_df.quantile(0.75))


    for coincidence, color in zip(COINCIDENCES, colors, ):
        plt.plot(range(tss_df.shape[0]), tss_df[coincidence],
                 label=coincidence, c=color)
        m = tss_df[coincidence].mean()
        plt.axhline(m, c="k", ls="dotted")
        plt.text(0.1, m, f"{coincidence} mean")

    for coincidence, quantile, minimum in \
        zip(COINCIDENCES, tss_top_quantiles, tss_mins):
        quantile_df = tss_df.loc[tss_df[coincidence] >= quantile]
        min_df = tss_df.loc[tss_df[coincidence] == minimum]

        plt.scatter(list(quantile_df.index.values), quantile_df[coincidence],
                    c="green",
                    label="top quartile" if coincidence == "all" else "")
        plt.scatter(list(min_df.index.values), min_df[coincidence],
                    c="darkviolet",
                    label="minimum" if coincidence == "all" else "")

    for shape, time_interval in zip(shapes, time_intervals):
        plt.axvline(shape - 1, c="k", ls="dashed")
        if time_interval == 24:
            plt.text(shape, 0.88, f"{time_interval}h Interval")
        else:
            plt.text(shape - 10, 0.9, f"{time_interval}h Interval")
    plt.title(get_title("TSS"))
    plt.legend()
    plt.xlabel("Average Time-point Index")
    plt.ylabel("True Skill Score")
    plt.savefig(
        f"{RESULTS_DIRECTORY}time_window_classification/figures/"
        f"nb_mx_mean_analysis_tss.png")
    plt.show()

    plt.clf()
    plt.subplots(figsize=(20, 13))
    for coincidence, color in zip(COINCIDENCES, colors):
        plt.plot(range(recall_df.shape[0]), recall_df[coincidence],
                 label=coincidence, c=color)
        m = tss_df[coincidence].mean()
        plt.axhline(m, c="k", ls="dotted")
        plt.text(0.1, m, f"{coincidence} mean")

    for coincidence, quantile, minimum in \
        zip(COINCIDENCES, recall_top_quantiles, recall_mins):
        quantile_df = recall_df.loc[recall_df[coincidence] >= quantile]
        min_df = recall_df.loc[recall_df[coincidence] == minimum]
        plt.scatter(list(quantile_df.index.values), quantile_df[coincidence],
                    c="green",
                    label="top quartile" if coincidence == "all" else "")
        plt.scatter(list(min_df.index.values), min_df[coincidence],
                    c="darkviolet",
                    label="minimum" if coincidence == "all" else "")

    for shape, time_interval in zip(shapes, time_intervals):
        plt.axvline(shape - 1, c="k", ls="dashed")
        if time_interval == 24:
            plt.text(shape, 0.88, f"{time_interval}h Interval")
        else:
            plt.text(shape - 10, 0.9, f"{time_interval}h Interval")
    plt.title(get_title("MX Recall"))
    plt.legend()
    plt.xlabel("Average Time-point Index")
    plt.ylabel("MX Recall")
    plt.savefig(
        f"{RESULTS_DIRECTORY}time_window_classification/figures/"
        f"nb_mx_mean_analysis_recall.png")
    plt.show()

