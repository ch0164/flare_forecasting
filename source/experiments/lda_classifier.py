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

    time_interval = 12
    lo_time = 10
    hi_time = lo_time + time_interval
    lo_time_complement = 24 - lo_time
    hi_time_complement = 24 - hi_time

    # Get the time window of the experiment for metadata.
    time_window = get_time_window(lo_time, hi_time)
    time_window_caption = time_window.replace("_", "-")

    # Obtain the properties for flares.
    flare_dataframes = [
        get_ar_properties(flare_class,
                          lo_time,
                          hi_time,
                          cleaned_data_directory,
                          now_string,
                          wipe_old_data=True,
                          use_time_window=True,
                          coincidence_time_window="0h_24h"
        )
        for flare_class in FLARE_CLASSES
    ]

    # Train and test LDA classifier with two classes: B and MX.
    _, all_b_df, all_mx_df = tuple(flare_dataframes)
    for coincidence in COINCIDENCES:
        all_flares_df = pd.concat([all_b_df, all_mx_df])
        # Group the flares by coincidence before splitting for training/testing.
        if coincidence == "coincident":
            all_flares_df = \
                all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
        elif coincidence == "noncoincident":
            all_flares_df = \
                all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]


        print(all_flares_df)

        train_size = 0.7
        test_size = 1 - train_size
        all_flares_df["xray_class"] = all_flares_df["xray_class"].apply(
            classify_flare, combine=True
        )
        all_flares_train_df, all_flares_test_df = \
            train_test_split(all_flares_df, test_size=test_size)
        all_flares_train_df.dropna(inplace=True)
        all_flares_test_df.dropna(inplace=True)
        train_target = all_flares_train_df["xray_class"]
        test_target = all_flares_test_df["xray_class"]

        train_lda = LinearDiscriminantAnalysis()
        train_components = train_lda.fit_transform(
            all_flares_train_df[FLARE_PROPERTIES], train_target)

        ld_labels = [f"LD{i + 1}" for i in range(1)]
        train_lda_df = pd.DataFrame(train_components, columns=ld_labels)
        train_lda_df["xray_class"] = list(train_target)

        train_b_df = train_lda_df.loc[train_lda_df["xray_class"] == "B"]
        train_b_count = train_b_df.shape[0]
        train_mx_df = train_lda_df.loc[train_lda_df["xray_class"] == "MX"]
        train_mx_count = train_mx_df.shape[0]

        b_centroid = train_b_df["LD1"].mean()
        mx_centroid = train_mx_df["LD1"].mean()
        midpoint = (b_centroid + mx_centroid) / 2

        test_lda = LinearDiscriminantAnalysis()
        test_components = test_lda.fit_transform(
            all_flares_test_df[FLARE_PROPERTIES], test_target)

        test_lda_df = pd.DataFrame(test_components, columns=ld_labels)
        test_lda_df["xray_class"] = list(test_target)

        test_b_df = test_lda_df.loc[test_lda_df["xray_class"] == "B"]
        test_b_count = test_b_df.shape[0]
        test_mx_df = test_lda_df.loc[test_lda_df["xray_class"] == "MX"]
        test_mx_count = test_mx_df.shape[0]

        # sns.set_palette(sns.color_palette(["#FF0B04", "#4374B3"]))
        fig, ax = plt.subplots()
        train_b_df["jitter"] = [random.uniform(0, 1) for _ in
                                range(train_b_df.shape[0])]
        train_mx_df["jitter"] = [random.uniform(0, 1) for _ in
                                 range(train_mx_df.shape[0])]
        test_b_df["jitter"] = [random.uniform(0, 1) for _ in
                               range(test_b_df.shape[0])]
        test_mx_df["jitter"] = [random.uniform(0, 1) for _ in
                                range(test_mx_df.shape[0])]

        train_b_df.plot(x="LD1", y="jitter", label="Train B",
                        kind="scatter", c="dodgerblue", ax=ax)
        train_mx_df.plot(x="LD1", y="jitter", label="Train MX",
                         kind="scatter", c="orangered", ax=ax)
        test_b_df.plot(x="LD1", y="jitter", label="Test B",
                       kind="scatter", c="darkblue", ax=ax)
        test_mx_df.plot(x="LD1", y="jitter", label="Test MX",
                        kind="scatter", c="darkred", ax=ax)

        ax.scatter([b_centroid], [0.5], color="k", marker='X')
        ax.scatter([mx_centroid], [0.5], color="k", marker='X')
        ax.scatter([midpoint], [0.5], color="k", marker='X')
        ax.axvline(x=midpoint, color="k")

        plt.title(f"{experiment_caption} on {coincidence.capitalize()} Flares "
                  f"{time_window_caption} Mean,\n"
                  f"Trained Using {train_size * 100}% of B/MX "
                  f"{coincidence.capitalize()} Flares")
        plt.tight_layout()
        plt.savefig(
            f"{figure_directory}{coincidence}_b_mx_scatterplot_"
            f"{time_window}_mean.png")
        plt.show()

        y_pred = ["B" if row["LD1"] <= midpoint else "MX"
                  for _, row in test_lda_df.iterrows()]

        tp, fn, fp, tn = confusion_matrix(test_target, y_pred).ravel()
        sensitivity = (tp / float(tp + fn))
        false_alarm_rate = (fp / float(fp + tn))
        tss = sensitivity - false_alarm_rate
        cr = classification_report(test_target, y_pred, output_dict=True)
        accuracy = cr["accuracy"]
        custom_cr = {
            "B": cr["B"],
            "MX": cr["MX"],
        }
        custom_cr["B"]["count"] = custom_cr["B"].pop("support")
        custom_cr["MX"]["count"] = custom_cr["MX"].pop("support")
        cr_df = pd.DataFrame(custom_cr).transpose()
        with open(f"{other_directory}{coincidence}_{time_window}"
                  f"_b_mx_classification_report.txt", "w") as f:
            stdout = sys.stdout
            sys.stdout = f
            print(f"{coincidence.capitalize()} Flares")
            print(f"Trained on {train_b_count} B Flares and "
                  f"{train_mx_count} MX Flares")
            print("-" * 50)
            print(cr_df)
            print()
            print(f"Accuracy: {accuracy}")
            print(f"TSS: {tss}")
            sys.stdout = stdout
        #
        # # calculate accuracy
        # conf_accuracy = (float(tp + tn) / float(tp + tn + fp + fn))
        #
        # # calculate the sensitivity

        # # calculate the specificity
        # conf_specificity = (tn / float(tn + fp))
        #

        #
        # # calculate precision
        # conf_precision = (tn / float(tn + fp))
        # # calculate f_1 score
        # conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
        # conf_tss = conf_sensitivity - conf_false_alarm_rate
        #
        # with open(f"{other_directory}b_mx_{time_window}_mean_classification_metrics_{now_string}.txt", "w", newline="\n") as f:
        #     f.write("Classification Metrics\n")
        #     f.write('-' * 50 + "\n")
        #     f.write(f"Total # of B Flares: {train_b_count}\n")
        #     f.write(f"Total # of MX Flares: {train_mx_count}\n")
        #     # f.write(f"Total # of NULL Flares: {test_null_count}\n")
        #     f.write('-' * 50 + "\n")
        #     f.write(f'Accuracy: {round(conf_accuracy, 2)}\n')
        #     f.write(f'B Recall/Sensitivity: {round(conf_sensitivity, 2)}\n')
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
