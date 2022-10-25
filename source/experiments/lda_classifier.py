################################################################################
# Filename: lda_classifier.py
# Description: This file classifies flares using LDA.
################################################################################
from copy import copy

import pandas as pd

# Custom Imports
from source.utilities import *
from scipy import stats
from num2words import num2words


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

    # Experiment Name (No Acronyms)
    experiment = "lda_classifier"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for today for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
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
    flare_dataframes = [
        get_ar_properties(flare_class,
                          lo_time,
                          hi_time,
                          coincidence_time_window="0h_24h",
                          coincidence_flare_classes="nbmx"
        )
        for flare_class in ["NB", "MX"]
    ]
    # flare_dataframes = []
    # for flare_class in FLARE_CLASSES:
    #     df = pd.read_csv(f"{cleaned_data_directory}{flare_class.lower()}_{time_window}_mean_dataset_20_09_2022_12_07_59.csv")
    #     flare_dataframes.append(df)

    # Plot specified flare properties over the specified time.
    # to_drop = ["d_l_f", "MEANGAM",
    #            "MEANGBH", "MEANGBT", "MEANGBZ", "MEANJZD",
    #            "MEANJZH", "slf"]
    # new_flare_properties = FLARE_PROPERTIES[:]
    # for prop in to_drop:
    #     new_flare_properties.remove(prop)

    # Train LDA classifier with two classes: B and MX.
    # Then, test LDA classifier with NULL flares.
    # null_df, train_b_df, train_mx_df = tuple(flare_dataframes)
    nb_df, mx_df = tuple(flare_dataframes)
    nb_df.dropna(inplace=True)
    mx_df.dropna(inplace=True)
    mx_df["xray_class"] = "MX"
    nb_df["xray_class"] = "NB"

    flares_list = [nb_df, mx_df]

    all_flares_df = pd.concat(flares_list).\
        reset_index().\
        drop(["index"], axis=1).\
        rename_axis("index")

    # Apply trimmed means.
    saved_df = all_flares_df
    plot = True
    for lda_index in range(2, 11):
        print(f"LOO LDA, Iteration {lda_index}")
        all_flares_df = saved_df.copy()
        test_df = pd.DataFrame()
        train_lda = LinearDiscriminantAnalysis()
        train_components = train_lda.fit_transform(all_flares_df[FLARE_PROPERTIES],
                                                   all_flares_df["xray_class"])
        train_lda_df = pd.DataFrame(train_components, columns=["LD1"])
        train_lda_df.index = all_flares_df.index
        train_lda_df["xray_class"] = pd.Series(all_flares_df["xray_class"])
        nb_df = train_lda_df.loc[
            train_lda_df["xray_class"] == "NB"].sort_values(by="LD1")
        mx_df = train_lda_df.loc[
            train_lda_df["xray_class"] == "MX"].sort_values(by="LD1")
        for _ in range(lda_index):
            n_to_drop = int(0.075 * nb_df.shape[0])
            nb_df = nb_df.iloc[:n_to_drop]
            n_to_drop = int(0.075 * mx_df.shape[0])
            mx_df = mx_df.iloc[mx_df.shape[0] - n_to_drop:]

            test_df = pd.concat([test_df, all_flares_df.iloc[nb_df.index]])
            test_df = pd.concat([test_df, all_flares_df.iloc[mx_df.index]])
            all_flares_df.drop(nb_df.index.values, inplace=True)
            all_flares_df.drop(mx_df.index.values, inplace=True)

            train_lda = LinearDiscriminantAnalysis()
            train_components = train_lda.fit_transform(
                all_flares_df[FLARE_PROPERTIES],
                all_flares_df["xray_class"])
            train_lda_df = pd.DataFrame(train_components, columns=["LD1"])
            # train_lda_df.index = all_flares_df.index
            if "level_0" in all_flares_df.columns:
                all_flares_df.drop("level_0", axis=1, inplace=True)
            all_flares_df.reset_index(inplace=True)
            train_lda_df["xray_class"] = pd.Series(all_flares_df["xray_class"])
            nb_df = train_lda_df.loc[
                train_lda_df["xray_class"] == "NB"].sort_values(by="LD1")
            mx_df = train_lda_df.loc[
                train_lda_df["xray_class"] == "MX"].sort_values(by="LD1")


        # all_flares_df.drop("level_0", axis=1, inplace=True)

        # lda_index = 0
        X = all_flares_df.drop("xray_class", axis=1)
        y = all_flares_df["xray_class"].to_numpy()
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        y_true, y_pred = [], []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            train_lda = LinearDiscriminantAnalysis()
            train_components = train_lda.fit_transform(X_train[FLARE_PROPERTIES], y_train)
            train_lda_df = pd.DataFrame(train_components, columns=["LD1"])
            train_lda_df["xray_class"] = pd.Series(y_train)

            train_nbc_df = train_lda_df.loc[train_lda_df["xray_class"] == "NB"]
            train_mx_df = train_lda_df.loc[train_lda_df["xray_class"] == "MX"]

            nbc_centroid = train_nbc_df["LD1"].mean()
            mx_centroid = train_mx_df["LD1"].mean()
            midpoint = (nbc_centroid + mx_centroid) / 2
            # midpoint += threshold

            # train_lda.intercept_ -= threshold

            pred = train_lda.predict(X_test[FLARE_PROPERTIES])[0]
            y_true.append(y_test[0])
            y_pred.append(pred)
            if 0 in test_index and plot:
                fig, ax = plt.subplots()
                # train_nb_df = train_lda_df.loc[train_lda_df["xray_class"] == "NB"]
                random.seed(10)
                train_nbc_df["jitter"] = [random.uniform(0, 1) for _ in range(train_nbc_df.shape[0])]
                train_mx_df["jitter"] = [random.uniform(0, 1) for _ in range(train_mx_df.shape[0])]
                # train_nbc_df["xray_class"] = nbc_labels
                # train_n_df = train_nbc_df.loc[train_nbc_df["xray_class"] == "N"]
                # train_b_df = train_nbc_df.loc[train_nbc_df["xray_class"] == "B"]
                # train_c_df = train_nbc_df.loc[train_nbc_df["xray_class"] == "C"]
                # train_mx_df["xray_class"] = mx_labels[1:]
                # train_m_df = train_mx_df.loc[train_mx_df["xray_class"] == "M"]
                # train_x_df = train_mx_df.loc[train_mx_df["xray_class"] == "X"]


                # train_n_df.plot(x="LD1", y="jitter", label="Train Null", kind="scatter", c="grey", ax=ax)
                # train_b_df.plot(x="LD1", y="jitter", label="Train B", kind="scatter", c="dodgerblue", ax=ax)
                # train_c_df.plot(x="LD1", y="jitter", label="Train C", kind="scatter", c="lightgreen", ax=ax)
                train_nbc_df.plot(x="LD1", y="jitter", label="Train NB", kind="scatter", c="dodgerblue", ax=ax)
                train_mx_df.plot(x="LD1", y="jitter", label="Train MX", kind="scatter", c="orangered", ax=ax)
                # train_m_df.plot(x="LD1", y="jitter", label="Train M", kind="scatter", c="orange", ax=ax)
                # train_x_df.plot(x="LD1", y="jitter", label="Train X", kind="scatter", c="red", ax=ax)
                ax.axvline(x=midpoint, color="k")
                ax.scatter([nbc_centroid], [0.5], color="k", marker='X')
                ax.scatter([mx_centroid], [0.5], color="k", marker='X')
                plt.title(f"{experiment_caption} LOO Testing, Training on NB and MX Flares\n"
                         f"from {time_window_caption} \n"
                          f"Trimmed Means ({0.075 * 100:.2f}% from Each Class), "
                          f"Iteration {lda_index}"
                          )
                fig.tight_layout()
                fig.savefig(f"{figure_directory}nb_mx_lda_loo_{time_window}_{num2words(lda_index)}_trimmed_means_other.png")
                fig.show()

        # midpoint = sum(midpoints) / len(midpoints)
        if not test_df.empty:
            y_true += list(test_df["xray_class"])
            y_pred += list(train_lda.predict(test_df[FLARE_PROPERTIES]))
        cm = confusion_matrix(y_true, y_pred, labels=["NB", "MX"])
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
        with open(f"{metrics_directory}nb_mx_loo_{time_window}_trimmed_means_other_{num2words(lda_index)}.txt", "w") as f:
            stdout = sys.stdout
            sys.stdout = f
            print("Confusion Matrix")
            print("-" * 50)
            print("  NB", " MX")
            print("NB", cm[0])
            print("MX ", cm[1])
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
