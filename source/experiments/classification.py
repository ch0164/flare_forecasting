################################################################################
# Filename: experiment_template.py
# Description: This file is a correlation for easy creation of new experiments.
################################################################################

# Custom Imports
import pandas as pd
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
    experiment = "classification"
    experiment_caption = experiment.title().replace("_", " ")

    lo_time = 10
    hi_time = 22
    flare_classes = ["B", "MX"]
    flare_class_caption = "_".join(flare_classes)
    random_state = 7
    cross_validation = "70_30_train_test"

    # ------------------------------------------------------------------------
    # Place any results in the directory for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    names = [
        "LDA",
        "QDA",
        "KNN 2",
        "KNN 3",
        "KNN 4",
        "NB",
        "Logistic Regression",
        "Linear SVM",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
    ]

    classifiers = [
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(2),
        KNeighborsClassifier(3),
        KNeighborsClassifier(4),
        GaussianNB(),
        LogisticRegression(),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
    ]

    time_window = f"{lo_time}h_{hi_time}h"
    time_window_caption = time_window.replace("_", "-")

    # Obtain the properties for flares.
    flare_dataframes = [
        get_ar_properties(flare_class, lo_time, hi_time,
                          coincidence_time_window="0h_24h").dropna()
        for flare_class in flare_classes
    ]
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

        all_flares_df = shuffle(all_flares_df, random_state=random_state)
        X = all_flares_df[FLARE_PROPERTIES].to_numpy()
        X = StandardScaler().fit_transform(X)
        y = all_flares_df["xray_class"].to_numpy()

        # loo = LeaveOneOut()
        # loo.get_n_splits(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )

        for name, clf in zip(names, classifiers):
            # y_true = []
            # y_pred = []
            # for train_index, test_index in loo.split(X):
                # X_train, X_test = X[train_index], X[test_index]
                # y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                # y_pred.append(clf.predict(X_test))
                # y_true.append(y_test)
                filename = f"{metrics_directory}{cross_validation}/{coincidence}/" \
                           f"{name.lower().replace(' ', '_')}_" \
                           f"b_mx_{time_window}_classification_metrics.txt"
                write_classification_metrics(y_test, y_pred, filename, name,
                                             flare_classes=flare_classes,
                                             print_output=False)


if __name__ == "__main__":
    main()
