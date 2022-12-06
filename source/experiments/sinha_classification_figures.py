################################################################################
# Filename: sinha_classification_figures.py
# Description:
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
    experiment = "sinha_classification"
    experiment_caption = experiment.title().replace("_", " ")

    lo_time = 5
    hi_time = 17
    flare_classes = ["BC", "MX"]
    flare_class_caption = "_".join(flare_classes)
    random_state = 7
    cross_validation = "leave_one_out"

    # ------------------------------------------------------------------------
    # Place any results in the directory for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    names = [
        "KNN",
        "Logistic Regression",
        # "Linear SVM",
        # "Random Forest",
    ]

    classifiers = [
        KNeighborsClassifier(),
        LogisticRegression(),
        # SVC(kernel="linear", C=0.025),
        # RandomForestClassifier(),
    ]

    parameters_list = [
        [i for i in range(1, 17)],  # KNN
        [1 * (10 ** i) for i in range(-4, 5)],  # LR
        # [10, 120, 230, 340, 450, 560, 670, 780, 890, 1000],  # RFC
        # [],
        # [],
        # [],
        # [],


    ]

    parameter_names = [
        "n_neighbors",  # KNN
        "C",  # LR
        # "n_estimators",  # RFC
        # [],
        # [],
        # [],
        # [],
    ]

    # for (lo_time, hi_time) in [(12, 13), (21, 23), (5, 17)]:
    lo_time = 0
    hi_time = 24

    time_window = f"{lo_time}h_{hi_time}h"
    time_window_caption = time_window.replace("_", "-")

    # Obtain the properties for flares.
    flare_dataframes = [
        get_ar_properties(flare_class, lo_time, hi_time,
                          coincidence_flare_classes="nbcmx").dropna()
        for flare_class in flare_classes
    ]

    for coincidence in ["all", "coincident", "noncoincident"]:
        if coincidence == "coincident":
            is_coincident = True
        elif coincidence == "noncoincident":
            is_coincident = False
        else:
            is_coincident = None
        all_flares_df = pd.concat(flare_dataframes).dropna()
        all_flares_df = all_flares_df.loc[all_flares_df["xray_class"] != "N"]
        if is_coincident is not None:
            all_flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == is_coincident]

        all_flares_df = all_flares_df. \
            reset_index(). \
            drop(["index", "level_0"], axis=1). \
            rename_axis("index")

        all_flares_df["xray_class"].replace("M", "MX", inplace=True)
        all_flares_df["xray_class"].replace("X", "MX", inplace=True)
        all_flares_df["xray_class"].replace("B", "BC", inplace=True)
        all_flares_df["xray_class"].replace("C", "BC", inplace=True)
        # all_flares_df["xray_class"].replace("B", "BC", inplace=True)
        # all_flares_df["xray_class"].replace("C", "BC", inplace=True)

        all_flares_df["xray_class"].replace("MX", 1, inplace=True)
        all_flares_df["xray_class"].replace("BC", 0, inplace=True)
        for name, clf, parameters, parameter_name in zip(names, classifiers, parameters_list, parameter_names):
            occurrences = {p: 0 for p in parameters}
            cv = GridSearchCV(clf, {parameter_name: parameters}, cv=10, scoring="recall")
            for shuffle_index in range(20):
                shuffle_data = shuffle(all_flares_df)
                X = shuffle_data[FLARE_PROPERTIES].to_numpy()
                X = MinMaxScaler().fit_transform(X)
                y = shuffle_data["xray_class"].to_numpy()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                cv.fit(X_train, y_train)
                occurrences[cv.best_params_[parameter_name]] += 1
                print(f"{shuffle_index}/20", cv.best_params_[parameter_name])
                if shuffle_index == 19:
                    clf.set_params(**{parameter_name: max(occurrences.values())})
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_true = y_test
                    filename = f"{metrics_directory}{coincidence}/{'_'.join(flare_classes).lower()}_" \
                               f"optimized_{name.lower().replace(' ', '_')}_" \
                               f"{'_'.join(flare_classes).lower()}_{time_window}.txt"
                    write_classification_metrics(y_true, y_pred, filename, name,
                                                 flare_classes=[0, 1],
                                                 print_output=False)

            print(occurrences)
            plt.bar(range(len(occurrences.keys())), occurrences.values())
            plt.title(f"{name} Optimization, BC/MX {coincidence.capitalize()} Flares, 0h-24h")
            plt.xlabel(f"{parameter_name}")
            plt.ylabel("Occurrences")
            plt.xticks(range(len(parameters)), labels=parameters, rotation=45)
            plt.yticks([i for i in range(max(occurrences.values()) + 1)])
            plt.tight_layout()
            plt.savefig(f"{figure_directory}{coincidence}/{'_'.join(flare_classes).lower()}_{name.replace(' ', '_').lower()}_histogram.png")
            plt.show()

    # filename = f"{metrics_directory}{cross_validation}/" \
    #            f"{name.lower().replace(' ', '_')}_" \
    #            f"{'_'.join(flare_classes).lower()}_{time_window}_anova_params_classification_metrics.txt"
    # write_classification_metrics(y_true, y_pred, filename, name,
    #                              flare_classes=flare_classes,
    #                              print_output=False)


if __name__ == "__main__":
    main()
