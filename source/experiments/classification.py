################################################################################
# Filename: experiment_template.py
# Description: This file is a correlation for easy creation of new experiments.
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
    experiment = "classification"
    experiment_caption = experiment.title().replace("_", " ")

    lo_time = 5
    hi_time = 17
    flare_classes = ["NB", "MX"]
    flare_class_caption = "_".join(flare_classes)
    random_state = 7
    cross_validation = "leave_one_out"

    # ------------------------------------------------------------------------
    # Place any results in the directory for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    names = [
        "LDA",
        # "QDA",
        # "KNN 2",
        "KNN 3",
        # "KNN 4",
        # "NB",
        "Logistic Regression",
        "Linear SVM",
        # "RBF SVM",
        # "Decision Tree",
        "Random Forest",
        # "Neural Net",
        # "AdaBoost",
    ]

    classifiers = [
        LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis(),
        # KNeighborsClassifier(2),
        KNeighborsClassifier(3),
        # KNeighborsClassifier(4),
        # GaussianNB(),
        LogisticRegression(),
        SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
    ]

    clf_scores = {name: [] for name in names}

    # for (lo_time, hi_time) in [(12, 13), (21, 23), (5, 17)]:
    for (lo_time, hi_time) in [(21, 23), (5, 17)]:
        time_window = f"{lo_time}h_{hi_time}h"
        time_window_caption = time_window.replace("_", "-")
        file = f"{RESULTS_DIRECTORY}correlation/other/" \
               f"nb_mx_anova_f_scores_{time_window}.csv"
        anova_df = pd.read_csv(file)

        # Obtain the properties for flares.
        flare_dataframes = [
            get_ar_properties(flare_class, lo_time, hi_time,
                              coincidence_flare_classes="nbmx").dropna()
            for flare_class in flare_classes
        ]
        all_flares_df = pd.concat(flare_dataframes)

        all_flares_df = all_flares_df. \
            reset_index(). \
            drop(["index", "level_0"], axis=1). \
            rename_axis("index")

        all_flares_df["xray_class"].replace("M", "MX", inplace=True)
        all_flares_df["xray_class"].replace("X", "MX", inplace=True)
        all_flares_df["xray_class"].replace("N", "NB", inplace=True)
        all_flares_df["xray_class"].replace("B", "NB", inplace=True)
        # all_flares_df["xray_class"].replace("B", "BC", inplace=True)
        # all_flares_df["xray_class"].replace("C", "BC", inplace=True)

        all_flares_df = shuffle(all_flares_df, random_state=random_state)
        y = all_flares_df["xray_class"].to_numpy()
        for k in range(20):
            params = anova_df.iloc[:k + 1]["parameter"].values
            X = all_flares_df[params].to_numpy()
            X = StandardScaler().fit_transform(X)
            # jitter = np.array([random.uniform(0, 1) for _ in range(len(X))])
            # X_jitter = np.insert(X, 1, jitter, axis=1)

            loo = LeaveOneOut()
            loo.get_n_splits(X)
            for name, clf in zip(names, classifiers):
                print(f"LOO on {name} Classifier, k={k+1}")
                y_true = []
                y_pred = []
                for train_index, test_index in loo.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    clf.fit(X_train, y_train)
                    y_pred.append(clf.predict(X_test))
                    y_true.append(y_test)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                detection_rate = tp / float(tp + fn)
                false_alarm_rate = fp / float(fp + tn)
                tss = detection_rate - false_alarm_rate
                clf_scores[name].append(tss)

            d = json.dumps(clf_scores, indent=4)
            with open(f"{other_directory}{time_window}_anova.json", "w") as f:
                f.write(d)
                # filename = f"{metrics_directory}{cross_validation}/" \
                #            f"{name.lower().replace(' ', '_')}_" \
                #            f"{'_'.join(flare_classes).lower()}_{time_window}_anova_params_classification_metrics.txt"
                # # y_true = y_test
                # write_classification_metrics(y_true, y_pred, filename, name,
                #                              flare_classes=flare_classes,
                #                              print_output=False)


if __name__ == "__main__":
    main()
