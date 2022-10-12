################################################################################
# Filename: experiment_template.py
# Description: This file is a correlation for easy creation of new experiments.
################################################################################

# Custom Imports
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, plot_roc_curve
from sklearn.preprocessing import StandardScaler
from copy import copy

from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "time_window_classification"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    names = [
        "LDA",
        # "KNN 3",
        # "Logistic Regression",
        # "Linear SVM",
        # "Random Forest",
    ]

    classifiers = [
        LinearDiscriminantAnalysis(),
        # KNeighborsClassifier(3),
        # LogisticRegression(),
        # SVC(kernel="linear", C=0.025),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    ]

    time_intervals = [1, 2, 4, 8, 12, 24]
    flare_classes = ["NB", "MX"]
    random_state = 7
    for time_interval in time_intervals:
        time_interval_caption = f"{time_interval}h"
        mx_recall_df = pd.DataFrame(columns=COINCIDENCES)
        tss_df = pd.DataFrame(columns=COINCIDENCES)
        possible_time_windows = [
            (start_time, end_time) for start_time in range(0, 24)
            for end_time in range(1, 25)
            if start_time < end_time and
               abs(end_time - start_time) == time_interval
        ]

        for index, (lo_time, hi_time) in enumerate(possible_time_windows):
            print(f"{time_interval}, {index}/{len(possible_time_windows)}")
            time_window = f"{lo_time}h_{hi_time}h"
            time_window_caption = time_window.replace("_", "-")

            # Obtain the properties for flares.
            flare_dataframes = [
                get_ar_properties(flare_class, lo_time, hi_time,
                                  coincidence_time_window="0h_24h",
                                  coincidence_flare_classes="nbmx").dropna()
                for flare_class in flare_classes
            ]
            print("test")
            for coincidence in ["all", "coincident", "noncoincident"]:
                all_flares_df = pd.concat(flare_dataframes)
                if coincidence == "coincident":
                    all_flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
                elif coincidence == "noncoincident":
                    all_flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]

                # all_flares_df = pd.concat(flare_dataframes). \
                #     reset_index(). \
                #     drop(["index"], axis=1). \
                #     rename_axis("index")

                all_flares_df = all_flares_df. \
                    reset_index(). \
                    drop(["index", "level_0"], axis=1). \
                    rename_axis("index")

                all_flares_df["xray_class"].replace("M", "MX", inplace=True)
                all_flares_df["xray_class"].replace("X", "MX", inplace=True)
                all_flares_df["xray_class"].replace("N", "NB", inplace=True)
                all_flares_df["xray_class"].replace("B", "NB", inplace=True)

                all_flares_df = shuffle(all_flares_df, random_state=random_state)
                y = all_flares_df["xray_class"].to_numpy()
                X = all_flares_df[FLARE_PROPERTIES]
                X = StandardScaler().fit_transform(X)

                loo = LeaveOneOut()
                loo.get_n_splits(X)
                for name, clf in zip(names, classifiers):
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
                    cr = classification_report(y_true, y_pred, labels=flare_classes, output_dict=True)
                    tss_df.loc[index, coincidence] = tss
                    mx_recall_df.loc[index, coincidence] = cr['MX']['recall']

            tss_df.to_csv(f"{metrics_directory}{time_interval_caption}_nb_mx_lda_tss.csv")
            mx_recall_df.to_csv(f"{metrics_directory}{time_interval_caption}_nb_mx_lda_recall.csv")



if __name__ == "__main__":
    main()
