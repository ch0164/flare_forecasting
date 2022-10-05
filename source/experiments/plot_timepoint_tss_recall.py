import pandas as pd
from matplotlib import pyplot as plt


from source.constants import RESULTS_DIRECTORY, COINCIDENCES


metrics = ["tss", "recall"]

if __name__ == "__main__":
    dir = RESULTS_DIRECTORY + "time_window_classification/metrics/"
    plt.subplots(figsize=(20, 13))
    tss_df = pd.read_csv(f"{dir}timepoint_b_mx_lda_tss.csv").iloc[::-1]
    recall_df = pd.read_csv(f"{dir}timepoint_b_mx_lda_recall.csv").iloc[::-1]


    colors = ["grey", "blue", "red"]

    def get_title(metric):
        return f"Timepoint Analysis, {metric} from LDA Classifier Using LOO CV, for B/MX Flares"

    for coincidence, color in zip(COINCIDENCES, colors):
        plt.plot(range(tss_df.shape[0]), tss_df[coincidence],
                 label=coincidence, c=color)
        m = tss_df[coincidence].mean()
        plt.axhline(m, c="k", ls="dotted")
        plt.text(0.1, m, f"{coincidence} mean")
    plt.title(get_title("TSS"))
    plt.legend()
    plt.xlabel("Timepoint Index")
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
    plt.title(get_title("MX Recall"))
    plt.legend()
    plt.xlabel("Timepoint Index")
    plt.ylabel("MX Recall")
    plt.show()

