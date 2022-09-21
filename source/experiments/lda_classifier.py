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
    center_time = 16
    time_interval = 12

    time_interval = 24
    lo_time = 0
    hi_time = lo_time + time_interval

    # Get the time window of the experiment for metadata.
    # lo_time = int((center_time - time_interval // 2))
    # hi_time = int((center_time + time_interval // 2))
    time_window = f"{lo_time}h_{hi_time}h"
    time_window_caption = time_window.replace("_", "-")

    # Obtain the properties for flares.
    # flare_dataframes = [
    #     get_ar_properties(flare_class,
    #                       lo_time,
    #                       hi_time,
    #                       cleaned_data_directory,
    #                       now_string,
    #                       wipe_old_data=False,
    #                       coincidence_time_window="0h_24h"
    #     )
    #     for flare_class in FLARE_CLASSES
    # ]
    flare_dataframes = []
    for flare_class in FLARE_CLASSES:
        df = pd.read_csv(f"{cleaned_data_directory}{flare_class.lower()}_{time_window}_mean_dataset_20_09_2022_12_07_59.csv")
        flare_dataframes.append(df)

    # Plot specified flare properties over the specified time.
    to_drop = ["MEANGBH", "MEANGBT", "MEANGBZ", "MEANJZD", "slf"]
    new_flare_properties = FLARE_PROPERTIES[:]
    for prop in to_drop:
        new_flare_properties.remove(prop)

    # Train LDA classifier with two classes: B and MX.
    # Then, test LDA classifier with NULL flares.
    # null_df, train_b_df, train_mx_df = tuple(flare_dataframes)
    null_df, b_df, mx_df = tuple(flare_dataframes)
    null_df["xray_class"] = "N"
    mx_df["xray_class"] = "MX"
    b_df = b_df.loc[b_df["xray_class"] == "B"]
    # train_b_df, train_mx_df = b_df, mx_df
    # print(null_df.dropna())

    all_flares_df = pd.concat([null_df, b_df, mx_df]).\
        reset_index().\
        drop(["index"], axis=1).\
        rename_axis("index").dropna()
    all_flares_df["xray_class"].replace("N", "NB", inplace=True)
    all_flares_df["xray_class"].replace("B", "NB", inplace=True)
    all_flares_df = shuffle(all_flares_df)
    X = all_flares_df.drop("xray_class", axis=1)
    y = all_flares_df["xray_class"].to_numpy()
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    y_true, y_pred = [], []
    i = 0
    midpoints = []
    for train_index, test_index in loo.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_lda = LinearDiscriminantAnalysis()
        train_components = train_lda.fit_transform(X_train[new_flare_properties], y_train)
        train_lda_df = pd.DataFrame(train_components, columns=["LD1"])
        train_lda_df["xray_class"] = pd.Series(y_train)

        nb_centroid = train_lda_df.loc[train_lda_df["xray_class"] == "NB"]["LD1"].mean()
        mx_centroid = train_lda_df.loc[train_lda_df["xray_class"] == "MX"]["LD1"].mean()
        midpoint = (nb_centroid + mx_centroid) / 2
        # midpoints.append(midpoint)

        pred = train_lda.predict(X_test[new_flare_properties])
        y_true.append(y_test)
        y_pred.append(pred)
        if 0 in test_index:
            fig, ax = plt.subplots()
            train_nb_df = train_lda_df.loc[train_lda_df["xray_class"] == "NB"]
            train_mx_df = train_lda_df.loc[train_lda_df["xray_class"] == "MX"]
            nb_centroid = train_nb_df["LD1"].mean()
            mx_centroid = train_mx_df["LD1"].mean()
            midpoint = (nb_centroid + mx_centroid) / 2
            train_nb_df["jitter"] = [random.uniform(0, 1) for _ in range(train_nb_df.shape[0])]
            train_mx_df["jitter"] = [random.uniform(0, 1) for _ in range(train_mx_df.shape[0])]

            train_nb_df.plot(x="LD1", y="jitter", label="Train NB", kind="scatter", c="dodgerblue", ax=ax)
            train_mx_df.plot(x="LD1", y="jitter", label="Train MX", kind="scatter", c="orangered", ax=ax)
            ax.axvline(x=midpoint, color="k")
            ax.scatter([nb_centroid], [0.5], color="k", marker='X')
            ax.scatter([mx_centroid], [0.5], color="k", marker='X')
            plt.title(f"{experiment_caption} LOO Testing, Training on Null/B and MX Flares\n"
                     f"from {time_window_caption}")
            fig.savefig(f"{figure_directory}nb_mx_lda_loo_{time_window}_{now_string}.png")
            fig.show()

    # midpoint = sum(midpoints) / len(midpoints)
    cm = confusion_matrix(y_true, y_pred)
    tp, fn, fp, tn = cm.ravel()
    cr = classification_report(y_true, y_pred, output_dict=True)
    accuracy = cr["accuracy"]
    sens = tp / float(tp + fn)
    far = fp / float(fp + tn)
    tss = sens - far
    custom_cr = {
        "NB": cr["NB"],
        "MX": cr["MX"],
    }
    custom_cr["NB"]["count"] = custom_cr["NB"].pop("support")
    custom_cr["MX"]["count"] = custom_cr["MX"].pop("support")
    cr_df = pd.DataFrame(custom_cr).transpose()
    with open(f"{other_directory}nb_mx_loo_{time_window}.txt", "w") as f:
        stdout = sys.stdout
        sys.stdout = f
        print("Confusion Matrix")
        print("-" * 50)
        print("   NB", "MX")
        print("NB", cm[0])
        print("MX", cm[1])
        print("-" * 50)
        print("Classification Metrics")
        print("-" * 50)
        print(cr_df)
        print()
        print(f"Accuracy: {accuracy}")
        print(f"TSS: {tss}")
        sys.stdout = stdout

    # b_test_size = 0.3
    # train_b_df, test_b_df = train_test_split(train_b_df, test_size=b_test_size)
    # train = pd.concat([train_b_df, train_mx_df]).dropna()
    # train_target = train["xray_class"]
    # test = pd.concat([null_df, test_b_df]).dropna()
    #
    # print(test.iloc[[55, 145, 194]].to_string())
    # exit(1)
    #
    # test_target = test["xray_class"]
    #
    # train_lda = LinearDiscriminantAnalysis()
    # train_components = train_lda.fit_transform(train[FLARE_PROPERTIES], train_target)
    #
    # ld_labels = [f"LD{i + 1}" for i in range(1)]
    # train_lda_df = pd.DataFrame(train_components, columns=ld_labels)
    # train_lda_df["xray_class"] = list(train_target)
    #
    # train_b_df = train_lda_df.loc[train_lda_df["xray_class"] == "B"]
    # train_b_count = train_b_df.shape[0]
    # train_mx_df = train_lda_df.loc[train_lda_df["xray_class"] == "MX"]
    # train_mx_count = train_mx_df.shape[0]
    #
    # b_centroid = train_b_df["LD1"].mean()
    # mx_centroid = train_mx_df["LD1"].mean()
    # midpoint = (b_centroid + mx_centroid) / 2
    #
    # test_lda = LinearDiscriminantAnalysis()
    # print(test["xray_class"])
    # test_components = test_lda.fit_transform(test[FLARE_PROPERTIES],
    #                                          test_target)
    #
    # ld_labels = [f"LD{i + 1}" for i in range(1)]
    # test_lda_df = pd.DataFrame(test_components, columns=ld_labels)
    # test_lda_df["xray_class"] = list(test["xray_class"])

    # sns.set_palette(sns.color_palette(["#FF0B04", "#4374B3"]))
    # fig, ax = plt.subplots()
    # train_b_df["jitter"] = [random.uniform(0, 1) for _ in range(train_b_df.shape[0])]
    # train_mx_df["jitter"] = [random.uniform(0, 1) for _ in range(train_mx_df.shape[0])]
    # test_lda_df["jitter"] = [random.uniform(0, 1) for _ in range(test_lda_df.shape[0])]
    #
    # test_b_df = test_lda_df.loc[test_lda_df["xray_class"] == "B"]
    # test_null_df = test_lda_df.loc[test_lda_df["xray_class"] == "N"]
    #
    # train_b_df.plot(x="LD1", y="jitter", label="Train B", kind="scatter", c="dodgerblue", ax=ax)
    # train_mx_df.plot(x="LD1", y="jitter", label="Train MX", kind="scatter", c="orangered", ax=ax)
    # test_b_df.plot(x="LD1", y="jitter", label="Test B", kind="scatter", c="darkblue", ax=ax)
    # test_null_df.plot(x="LD1", y="jitter", label="Test Null", kind="scatter", c="grey", ax=ax)
    #
    # ax.scatter([b_centroid], [0.5], color="k", marker='X')
    # ax.scatter([mx_centroid], [0.5], color="k", marker='X')
    # ax.scatter([midpoint], [0.5], color="k", marker='X')
    # ax.axvline(x=midpoint, color="k")
    #
    # y_pred = []
    # special = []
    # for index, row in test_lda_df.iterrows():
    #     x = row["LD1"]
    #     if x >= 3:
    #         special.append(index)
    #     if x <= midpoint:
    #         # y_pred.append("B")
    #         y_pred.append("NB")
    #     else:
    #         y_pred.append("MX")
    #
    # exit(1)
    #
    # target = ["NB" for _ in range(len(y_pred))]
    #
    # correct = len([y_pred[i] for i in range(len(y_pred)) if y_pred[i] == target[i]])
    # accuracy = correct / len(y_pred)
    #
    # plt.title(f"{experiment_caption} on Null/B Flares {time_window_caption} Mean\n"
    #           f"Trained by {b_test_size * 100}% B and 100% MX Flares with {accuracy * 100 :.2f}% NB Accuracy")
    # plt.tight_layout()
    # plt.savefig(
    #     f"{figure_directory}null_b_mx_scatterplot_{time_window}_mean_{now_string}.png")
    # plt.show()




    # with open(f"{other_directory}nb_mx_{time_window}_mean_classification_metrics_{now_string}.txt", "w", newline="\n") as f:
    #     f.write("Classification Metrics\n")
    #     f.write(f"Trained on {train_b_df} B Flares and {train_mx_count} MX Flares\n")
    #     f.write('-' * 50 + "\n")
    #     f.write(f'Accuracy: {round(conf_accuracy, 2)}\n')
    #     f.write(f'NB Recall/Sensitivity: {round(conf_sensitivity, 2)}\n')
    #     f.write(f'MX Recall/Specificity: {round(conf_specificity, 2)}\n')
    #     f.write(f'Precision: {round(conf_precision, 2)}\n')
    #     f.write(f'False Alarm Rate: {round(conf_false_alarm_rate, 2)}\n')
    #     f.write(f'f_1 Score: {round(conf_f1, 2)}\n')
    #     f.write(f"TSS: {round(conf_tss, 2)}\n")





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
