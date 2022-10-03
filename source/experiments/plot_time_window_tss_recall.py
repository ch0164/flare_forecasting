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
    for time_interval in time_intervals:
        tss_file = f"{dir}{time_interval}h_b_mx_lda_tss.csv"
        tss_df = pd.concat([tss_df, pd.read_csv(tss_file)])
        recall_file = f"{dir}{time_interval}h_b_mx_lda_recall.csv"
        recall_df = pd.concat([recall_df, pd.read_csv(recall_file)])
        shapes.append(recall_df.shape[0])

    colors = ["grey", "blue", "red"]

    def get_title(metric):
        return f"Time Window Mean Analysis, {metric}, for B/MX Flares"

    for coincidence, color in zip(COINCIDENCES, colors):
        plt.plot(range(tss_df.shape[0]), tss_df[coincidence],
                 label=coincidence, c=color)
        m = tss_df[coincidence].mean()
        plt.axhline(m, c="k", ls="dotted")
        plt.text(0.1, m, f"{coincidence} mean")
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
    plt.show()

    plt.clf()
    plt.subplots(figsize=(20, 13))
    for coincidence, color in zip(COINCIDENCES, colors):
        plt.plot(range(recall_df.shape[0]), recall_df[coincidence],
                 label=coincidence, c=color)
        m = tss_df[coincidence].mean()
        plt.axhline(m, c="k", ls="dotted")
        plt.text(0.1, m, f"{coincidence} mean")
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
    plt.show()

