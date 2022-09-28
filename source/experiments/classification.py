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
from sklearn.inspection import DecisionBoundaryDisplay
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
    cross_validation = "leave_one_out"

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
        # all_flares_df["xray_class"].replace("B", "BC", inplace=True)
        # all_flares_df["xray_class"].replace("C", "BC", inplace=True)

        all_flares_df = shuffle(all_flares_df, random_state=random_state)
        y = all_flares_df["xray_class"].to_numpy()
        X = all_flares_df[FLARE_PROPERTIES]
        X = StandardScaler().fit_transform(X)
        X = LinearDiscriminantAnalysis().fit_transform(X, y)
        jitter = np.array([random.uniform(0, 1) for _ in range(len(X))])
        X_jitter = np.insert(X, 1, jitter, axis=1)

        # loo = LeaveOneOut()
        # loo.get_n_splits(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )

        # Plot LDA reduced data points.
        fig, ax = plt.subplots(2, 7, figsize=(20, 15))
        i, j = 0, 0

        x_min, x_max = X_jitter[:, 0].min() - 0.5, X_jitter[:, 0].max() + 0.5
        y_min, y_max = X_jitter[:, 1].min() - 0.1, X_jitter[:, 1].max() + 0.1

        # Plot the training points
        train_df = pd.DataFrame(X_jitter, columns=["LD1", "jitter"])
        train_df["xray_class"] = y
        train_mx_df = train_df.loc[train_df["xray_class"] == "MX"]
        train_b_df = train_df.loc[train_df["xray_class"] != "MX"]
        train_mx_df.plot(x="LD1", y="jitter", label="MX", kind="scatter", c="orangered", ax=ax[i, j])
        train_b_df.plot(x="LD1", y="jitter", label="B", kind="scatter", c="dodgerblue", ax=ax[i, j])
        ax[i, j].set_xlim(x_min, x_max)
        ax[i, j].set_ylim(y_min, y_max)
        ax[i, j].set_title("LDA Reduced Data")

        for name, clf in zip(names, classifiers):
            # y_true = []
            # y_pred = []
            # for train_index, test_index in loo.split(X):
            #     X_train, X_test = X[train_index], X[test_index]
            #     y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )
            #     y_pred.append(clf.predict(X_test))
            #     y_true.append(y_test)
            filename = f"{metrics_directory}{cross_validation}/{coincidence}/" \
                       f"{name.lower().replace(' ', '_')}_" \
                       f"b_mx_lda_{time_window}_classification_metrics.txt"
            # y_true = y_test
            # write_classification_metrics(y_true, y_pred, filename, name,
            #                              flare_classes=flare_classes,
            #                              print_output=False)


if __name__ == "__main__":
    main()
