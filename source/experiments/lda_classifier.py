################################################################################
# Filename: lda_classifier.py
# Description: This file classifies flares using LDA.
################################################################################

# Custom Imports
from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "lda_classifier"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for today for the current experiment.
    now_string, cleaned_data_directory, figure_directory, other_directory = \
        build_experiment_directories(experiment)

    # ------------------------------------------------------------------------
    # Pre-process the data for the experiment.
    # Afterwards, place this data in `cleaned_data_directory`.
    needed_where = 8
    average_over = 12

    needed_flare_classes = FLARE_CLASSES[1:]
    print(needed_flare_classes)
    # flare_dataframes = [
    #     get_ar_properties(flare_class, needed_where, average_over)
    #     for flare_class in needed_flare_classes
    # ]

    # Get the time window of the experiment for metadata.
    lo_time = int(24 - (needed_where + average_over // 2))
    hi_time = int(24 - (needed_where - average_over // 2))
    time_window = f"{lo_time}h_{hi_time}h"
    time_window_caption = time_window.replace("_", "-")

    # # Save our original data.
    # for flare_class, df in zip(needed_flare_classes, flare_dataframes):
    #     df.to_csv(f"{cleaned_data_directory}{flare_class.lower()}_"
    #               f"{time_window}_mean_dataset_{now_string}.csv")

    # Perform LDA with two classes: B and MX.
    b_df, mx_df = \
        pd.read_csv(f"{cleaned_data_directory}b_10h_22h_mean_dataset_07_09_2022_13_42_35.csv"), \
        pd.read_csv(f"{cleaned_data_directory}mx_10h_22h_mean_dataset_07_09_2022_13_42_35.csv")
    b_df = b_df.loc[b_df["xray_class"] == "B"]
    mx_df["xray_class"] = "MX"
    bmx_df = pd.concat([b_df, mx_df]).dropna()

    # target = bmx_df["xray_class"]
    #
    # train, test = train_test_split(bmx_df, test_size=0.0)

    lda = LinearDiscriminantAnalysis()
    components = lda.fit_transform(bmx_df[FLARE_PROPERTIES], bmx_df["xray_class"])

    ld_labels = [f"LD{i + 1}" for i in range(1)]
    lda_df = pd.DataFrame(components, columns=ld_labels)
    lda_df["xray_class"] = list(bmx_df["xray_class"])

    b_df = lda_df.loc[lda_df["xray_class"] == "B"]
    b_count = b_df.shape[0]
    mx_df = lda_df.loc[lda_df["xray_class"] == "MX"]
    mx_count = mx_df.shape[0]

    b_centroid = b_df["LD1"].mean()
    mx_centroid = mx_df["LD1"].mean()
    midpoint = (b_centroid + mx_centroid) / 2

    # test_lda = LinearDiscriminantAnalysis()
    # test_components = test_lda.fit_transform(test[FLARE_PROPERTIES], test["xray_class"])
    #
    # ld_labels = [f"LD{i + 1}" for i in range(1)]
    # test_lda_df = pd.DataFrame(test_components, columns=ld_labels)
    # test_lda_df["xray_class"] = list(test["xray_class"])

    # b_df = test_lda_df.loc[test_lda_df["xray_class"] == "B"]
    # mx_df = test_lda_df.loc[test_lda_df["xray_class"] == "MX"]

    b_df = lda_df.loc[lda_df["xray_class"] == "B"]
    mx_df = lda_df.loc[lda_df["xray_class"] == "MX"]

    # sns.set_palette(sns.color_palette(["#FF0B04", "#4374B3"]))
    fig, ax = plt.subplots()
    b_df["jitter"] = [random.uniform(0, 1) for _ in range(b_df.shape[0])]
    mx_df["jitter"] = [random.uniform(0, 1) for _ in range(mx_df.shape[0])]

    b_df.to_csv(f"{cleaned_data_directory}b_{time_window}_mean_lda_{now_string}.csv")
    mx_df.to_csv(f"{cleaned_data_directory}mx_{time_window}_mean_lda_{now_string}.csv")
    b_df.plot(x="LD1", y="jitter", label="B", kind="scatter", c="dodgerblue", ax=ax)
    mx_df.plot(x="LD1", y="jitter", label="MX", kind="scatter", c="orangered", ax=ax)

    ax.scatter([b_centroid], [0.5], color="k", marker='X')
    ax.scatter([mx_centroid], [0.5], color="k", marker='X')
    ax.scatter([midpoint], [0.5], color="k", marker='X')
    ax.axvline(x=midpoint, color="k")

    plt.title(f"{experiment_caption} on B/MX {time_window_caption} Mean")
    plt.tight_layout()
    plt.savefig(
        f"{figure_directory}b_mx_scatterplot_{time_window}_mean_{now_string}.png")
    plt.show()

    y_pred = []
    df = pd.concat([b_df, mx_df])
    for index, row in df.iterrows():
        x = row["LD1"]
        if x <= midpoint:
            y_pred.append("B")
        else:
            y_pred.append("MX")

    from sklearn.metrics import confusion_matrix
    tp, fn, fp, tn = confusion_matrix(bmx_df["xray_class"], y_pred).ravel()

    # calculate accuracy
    conf_accuracy = (float(tp + tn) / float(tp + tn + fp + fn))

    # calculate the sensitivity
    conf_sensitivity = (tp / float(tp + fn))
    # calculate the specificity
    conf_specificity = (tn / float(tn + fp))

    conf_false_alarm_rate = (fp / float(fp + tn))

    # calculate precision
    conf_precision = (tn / float(tn + fp))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    conf_tss = conf_sensitivity - conf_false_alarm_rate

    with open(f"{other_directory}b_mx_{time_window}_mean_classification_metrics_{now_string}.txt", "w", newline="\n") as f:
        f.write("Classification Metrics\n")
        f.write('-' * 50 + "\n")
        f.write(f"Total # of B Flares: {b_count}\n")
        f.write(f"Total # of MX Flares: {mx_count}\n")
        f.write("Training Size: 0.7\n")
        f.write("Testing Size: 0.3\n")
        f.write('-' * 50 + "\n")
        f.write(f'Accuracy: {round(conf_accuracy, 2)}\n')
        f.write(f'B Recall/Sensitivity: {round(conf_sensitivity, 2)}\n')
        f.write(f'MX Recall/Specificity: {round(conf_specificity, 2)}\n')
        f.write(f'Precision: {round(conf_precision, 2)}\n')
        f.write(f'False Alarm Rate: {round(conf_false_alarm_rate, 2)}\n')
        f.write(f'f_1 Score: {round(conf_f1, 2)}\n')
        f.write(f"TSS: {round(conf_tss, 2)}\n")





    # ------------------------------------------------------------------------
    # Generate the figures of this experiment.
    # Afterwards, place these figures in `figure_directory`.
    # --- [ENTER CODE HERE]

    # ------------------------------------------------------------------------
    # Generate other kinds of output for the experiment.
    # Afterwards, place the output in `other_directory`.
    # --- [ENTER CODE HERE]


if __name__ == "__main__":
    main()
