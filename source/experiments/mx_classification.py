################################################################################
# Filename: mx_classification.py
# Description: TODO
################################################################################

# Custom Imports
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, plot_roc_curve
from sklearn.preprocessing import StandardScaler
import json
from copy import copy

from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "mx_classification"
    experiment_caption = experiment.title().replace("_", " ")

    lo_time = 5
    hi_time = 17
    flare_classes = ["NB", "MX"]
    flare_class_caption = "_".join(flare_classes).lower()
    random_state = 7
    time_window = f"{lo_time}h_{hi_time}h"
    time_window_caption = time_window.replace("_", "-")
    mx_classfied_by = "mx_classified_by"

    # ------------------------------------------------------------------------
    # Place any results in the directory for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    names = [
        "LDA",
        "KNN 3",
        "Logistic Regression",
        "Linear SVM",
        "Random Forest",
    ]

    classifiers = [
        LinearDiscriminantAnalysis(),
        KNeighborsClassifier(n_neighbors=3),
        LogisticRegression(C=1000),
        SVC(kernel="linear", C=100, gamma=0.001),
        RandomForestClassifier(n_estimators=120),
    ]

    clf_scores = {name: [] for name in names}

    flare_dataframes = [
        get_ar_properties(flare_class, lo_time, hi_time,
                          coincidence_time_window="0h_24h",
                          coincidence_flare_classes="nbmx").dropna()
        for flare_class in flare_classes
    ]
    all_flares_df = pd.concat(flare_dataframes)
    all_flares_df = all_flares_df. \
        reset_index(). \
        drop(["index"], axis=1). \
        rename_axis("index")

    labeled_flare_df = all_flares_df.copy()
    all_flares_df["xray_class"].replace("M", "MX", inplace=True)
    all_flares_df["xray_class"].replace("X", "MX", inplace=True)
    all_flares_df["xray_class"].replace("N", "NB", inplace=True)
    all_flares_df["xray_class"].replace("B", "NB", inplace=True)

    X = all_flares_df[FLARE_PROPERTIES]
    X = MinMaxScaler().fit_transform(X)
    y = all_flares_df["xray_class"].to_numpy()

    loo = LeaveOneOut()
    loo.get_n_splits(X)
    for name, clf in zip(names, classifiers):
        y_true = []
        y_pred = []
        train_mx = []
        additional_mx = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)[0]
            truth = y_test[0]
            y_pred.append(pred)
            y_true.append(truth)
            if pred == truth == "MX":
                train_mx.append(test_index[0])
            elif truth == "MX":
                additional_mx.append(test_index[0])

        # Write classification metrics of NB/MX.
        # filename = f"{metrics_directory}/" \
        #            f"{name.lower().replace(' ', '_')}_" \
        #            f"{flare_class_caption}_{time_window}_classification_metrics.txt"
        # write_classification_metrics(y_true, y_pred, filename, name,
        #                              flare_classes=flare_classes,
        #                              print_output=False)

        # Now, classify on "true" MXs.
        true_mx_df = labeled_flare_df.iloc[train_mx]
        X2 = true_mx_df[FLARE_PROPERTIES]
        X2 = MinMaxScaler().fit_transform(X2)
        y2 = true_mx_df["xray_class"].to_numpy()
        loo2 = LeaveOneOut()
        loo2.get_n_splits(X2)
        for name2, clf2 in zip(names, classifiers):
            y_true2 = []
            y_pred2 = []
            for train_index, test_index in loo.split(X2):
                X_train, X_test = X2[train_index], X2[test_index]
                y_train, y_test = y2[train_index], y2[test_index]
                clf2.fit(X_train, y_train)
                pred = clf2.predict(X_test)[0]
                truth = y_test[0]
                y_pred2.append(pred)
                y_true2.append(truth)

            y_true2 += list(labeled_flare_df.iloc[additional_mx]["xray_class"])
            y_pred2 += list(clf2.predict(labeled_flare_df.iloc[additional_mx][FLARE_PROPERTIES]))
            filename2 = f"{metrics_directory}/{mx_classfied_by}_{name.lower().replace(' ', '_')}/" \
                       f"{name2.lower().replace(' ', '_')}_" \
                       f"representative_mx_{time_window}_classification_metrics.txt"
            write_classification_metrics(y_true2, y_pred2, filename2, name2,
                                         flare_classes=["M", "X"],
                                         print_output=False)


if __name__ == "__main__":
    main()
