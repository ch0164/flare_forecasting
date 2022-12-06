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
from sklearn.metrics import RocCurveDisplay, plot_roc_curve, \
    ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import json
from copy import copy

from source.utilities import *


SINHA_PARAMETERS = [
    "TOTUSJH",
    "USFLUX",
    "TOTUSJZ",
    "R_VALUE",
    "TOTPOT",
    "AREA_ACR",
    "SAVNCPP",
    "ABSNJZH",
    "MEANPOT",
    "SHRGT45",
]

def main() -> None:
    plt.style.use('dark_background')
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
        # "LDA",
        # "QDA",
        # "KNN 2",
        # "KNN 3",
        # "KNN 4",
        # "NB",
        "Logistic Regression",
        # "Linear SVM",
        # "RBF SVM",
        # "Decision Tree",
        # "Random Forest",
        # "Neural Net",
        # "AdaBoost",
    ]

    classifiers = [
        # LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis(),
        # KNeighborsClassifier(2),
        # KNeighborsClassifier(3),
        # KNeighborsClassifier(4),
        # GaussianNB(),
        LogisticRegression(),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
    ]

    clf_scores = {name: [] for name in names}

    # for (lo_time, hi_time) in [(12, 13), (21, 23), (5, 17)]:
    fig, ax = plt.subplots(ncols=3, figsize=(8, 4))
    lo_time, hi_time = 0, 24
        # Obtain the properties for flares.
        # flare_dataframes = [
        #     get_ar_properties(flare_class, lo_time, hi_time, filter_multiple_ars=False,
        #                       coincidence_flare_classes="nbmx")
        #     for flare_class in ["NB", "MX"]
        # ]
        # all_flares_df = pd.concat(flare_dataframes)

    all_flares_df = pd.read_csv("agu_dataset.csv")

    all_flares_df["xray_class"].replace("M", "MX", inplace=True)
    all_flares_df["xray_class"].replace("X", "MX", inplace=True)
    # all_flares_df["xray_class"].replace("N", "NB", inplace=True)
    all_flares_df = all_flares_df.loc[all_flares_df["xray_class"] != "N"]
    print(all_flares_df["xray_class"])
    for param in SINHA_PARAMETERS:
        all_flares_df[param] = all_flares_df[param].fillna(all_flares_df[param].mean())

    temp_df = all_flares_df.copy()
    for axis_index, coincidence in enumerate(COINCIDENCES):
        if coincidence == "coincident":
            all_flares_df = temp_df.loc[temp_df["COINCIDENCE"] == True]
        elif coincidence == "noncoincident":
            all_flares_df = temp_df.loc[temp_df["COINCIDENCE"] == False]
        else:
            all_flares_df = temp_df
        y = all_flares_df["xray_class"].to_numpy()
        X = all_flares_df[SINHA_PARAMETERS].to_numpy()
        X = StandardScaler().fit_transform(X)

        loo = LeaveOneOut()
        loo.get_n_splits(X)
        for name, clf in zip(names, classifiers):
            y_true = []
            y_pred = []
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred.append(clf.predict(X_test))
                y_true.append(y_test)
            cm = confusion_matrix(y_true, y_pred)
            print(cm)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            detection_rate = tp / float(tp + fn)
            false_alarm_rate = fp / float(fp + tn)
            tss = detection_rate - false_alarm_rate
            clf_scores[name].append(tss)

            write_classification_metrics(y_true, y_pred, name,
                                         flare_classes=["B", "MX"],
                                         print_output=True)

            cm, cm_label = confusion_matrix(y_true, y_pred), coincidence
            write_classification_metrics(y_true, y_pred,
                                         "LDA-based Classifier",
                                         ["B", "MX"], print_output=True)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=["B", "MX"])
            disp.plot(ax=ax[axis_index], colorbar=False)
            tn, fp, fn, tp = cm.ravel()
            detection_rate = tp / float(tp + fn)
            false_alarm_rate = fp / float(fp + tn)
            tss = detection_rate - false_alarm_rate
            disp.ax_.set_title(f"{cm_label.capitalize()} (TSS: {tss:.2f})")
            disp.ax_.set_xlabel("")
            if axis_index != 0:
                disp.ax_.set_ylabel("")
            # disp.im_.colorbar.remove()

    fig.text(0.5, 0.1, "Predicted label", ha='center')
    plt.suptitle("LDA-based Classifier, Confusion Matrices")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
