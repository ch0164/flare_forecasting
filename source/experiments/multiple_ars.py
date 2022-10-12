################################################################################
# Filename: multiple_ars.py
# Description: Todo
################################################################################

# Custom Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler

from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "multiple_ars"
    experiment_caption = experiment.title().replace("_", " ")

    names = [
        "lda",
        "linear_svm",
    ]

    classifiers = [
        LinearDiscriminantAnalysis(),
        SVC(kernel="linear", C=0.025),
    ]

    # Place any results in the directory for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    flare_classes = ["NB", "MX"]
    lo_time = 0
    hi_time = 24
    flare_dataframes = [
        get_ar_properties(flare_class, lo_time, hi_time,
                          coincidence_flare_classes="nbmx",
                          filter_multiple_ars=False)
        for flare_class in flare_classes
    ]
    all_flares_df = pd.concat(flare_dataframes).dropna()
    all_flares_df["xray_class"].replace("M", "MX", inplace=True)
    all_flares_df["xray_class"].replace("X", "MX", inplace=True)
    all_flares_df["xray_class"].replace("N", "NB", inplace=True)
    all_flares_df["xray_class"].replace("B", "NB", inplace=True)
    all_flares_df = shuffle(all_flares_df, random_state=7)

    occurrences = ["singular", "multiple"]
    mx_recall_df = pd.DataFrame(columns=occurrences)
    tss_df = pd.DataFrame(columns=occurrences)
    for occurrence in occurrences:
        ars_df = all_flares_df[all_flares_df["ARs"] == occurrence]
        y = ars_df["xray_class"].to_numpy()
        X = ars_df[FLARE_PROPERTIES]
        X = StandardScaler().fit_transform(X)
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        for index, (name, clf) in enumerate(zip(names, classifiers)):
            y_pred, y_true = [], []
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_true.append(y_test)
                y_pred.append(clf.predict(X_test))

            cm = confusion_matrix(y_true, y_pred, labels=flare_classes)
            tn, fp, fn, tp = cm.ravel()
            detection_rate = tp / float(tp + fn)
            false_alarm_rate = fp / float(fp + tn)
            tss = detection_rate - false_alarm_rate
            cr = classification_report(y_true, y_pred, labels=flare_classes,
                                       output_dict=True)
            tss_df.loc[index, occurrence] = tss
            mx_recall_df.loc[index, occurrence] = cr['MX']['recall']

    tss_df.index = names
    mx_recall_df.index = names
    tss_df.to_csv(
        f"{metrics_directory}nb_mx_multiple_ars_lda_svm_tss.csv")
    mx_recall_df.to_csv(
        f"{metrics_directory}nb_mx_multiple_ars_lda_svm_recall.csv")

    # Uncomment below to print counts.
    # for time_frame in ["All Time", "Solar Cycle 24"]:
    #     counts_df = pd.DataFrame(columns=[
    #         "total",
    #         "singular",
    #         "multiple",
    #         "singular_all",
    #         "singular_coincident",
    #         "singular_noncoincident",
    #         "multiple_all",
    #         "multiple_coincident",
    #         "multiple_noncoincident",
    #     ])
    #     df = pd.DataFrame()
    #     for index, flare_class in enumerate(["B", "N", "M", "X"]):
    #         df = all_flares_df.loc[
    #             (all_flares_df["xray_class"] == flare_class)
    #         ]
    #         if time_frame == "Solar Cycle 24":
    #             df = df.loc[
    #                 (df["time_start"].str.contains("2013")) |
    #                 (df["time_start"].str.contains("2014"))
    #             ]
    #         counts_df.at[index, "total"] = df.shape[0]
    #         for occurrence in ["singular", "multiple"]:
    #             df = all_flares_df.loc[
    #                 (all_flares_df["xray_class"] == flare_class) &
    #                 (all_flares_df["ARs"] == occurrence)
    #                 ]
    #             counts_df.at[index, occurrence] = df.shape[0]
    #             for coincidence in COINCIDENCES:
    #                 temp_df = df
    #                 if coincidence == "coincident":
    #                     temp_df = df.loc[df["COINCIDENCE"] == True]
    #                 elif coincidence == "noncoincident":
    #                     temp_df = df.loc[df["COINCIDENCE"] == False]
    #                 counts_df.at[index, f"{occurrence}_{coincidence}"] = temp_df.shape[0]
    #     counts_df.index = ["B", "N", "M", "X"]
    #     counts_df.to_csv(f"{metrics_directory}nb_mx_counts_{time_frame}.csv")


if __name__ == "__main__":
    main()
