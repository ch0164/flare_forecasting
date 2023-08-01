################################################################################
# Filename: sinha_study.py
################################################################################

# Custom Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import json

from source.utilities import *

# Disable Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Experiment Name (No Acronyms)
experiment = "sinha_study"
experiment_caption = experiment.title().replace("_", " ")

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

singh_filename = r"C:\Users\youar\PycharmProjects\flare_forecasting\flare_data\singh_nabmx_24h_default_timepoint_without_filter.csv"

# SINHA_PARAMETERS = [
#     "TOTUSJH",
#     "TOTPOT",
#     "TOTUSJZ",
#     "ABSNJZH",
#     "SAVNCPP",
#     "USFLUX",
#     "AREA_ACR",
#     "MEANPOT",
#     "R_VALUE",
#     "SHRGT45",
#     "EPSZ",
#     "TOTBSQ",
#     "TOTFZ",
#     "TOTABSTWIST"
# ]

score_names = [
    "TSS",
    "MAC",
    "SSW",
    "CSW",
]

names = [
        "KNN",
        "RFC",
        "LR",
        "SVM",
]

# classifiers = [
#     KNeighborsClassifier(),
#     RandomForestClassifier(),
#     LogisticRegression(),
#     SVC(),
# ]

# sinha
classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    RandomForestClassifier(n_estimators=120),
    LogisticRegression(C=1000),
    SVC(C=100, gamma=0.001),
]


max_scores_dict = {name: {score: 0 for score in score_names} for name in names}
random_states_dict = {name: {score: 0 for score in score_names} for name in names}


parameters = [
    dict(n_neighbors=list(range(1, 16 + 1))),
    dict(n_estimators=[10 + 110 * i for i in range(10)]),
    dict(C=[10**e for e in range(-5, 5)]),
    dict(C=[10**e for e in range(-4, 3)],
         gamma=[10**e for e in range(-5, 2)]),
]

param_names = [
    "n_neighbors",
    "n_estimators",
    "C",
    ("C", "gamma")
]

occurrences = {
    "KNN": [],
    "RFC": [],
    "LR": [],
    "SVM": [],
}

# ------------------------------------------------------------------------
# Place any results in the directory for the current experiment.
now_string, figure_directory, metrics_directory, other_directory = \
    build_experiment_directories(experiment)


def get_tss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    detection_rate = tp / float(tp + fn)
    false_alarm_rate = fp / float(fp + tn)
    tss = detection_rate - false_alarm_rate
    return tss


def get_mac(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    detection_rate = tp / float(tp + fn)
    true_negative_rate = tn / float(fp + tn)
    mac = 0.5 * (detection_rate + true_negative_rate)
    return mac


def get_ssw(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    ssw = (tp - fn) / (tp + fn)
    return ssw


def get_csw(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    csw = (tn - fp) / (tn + fp)
    return csw


def floor_minute(time, cadence=12):
    import datetime
    if not isinstance(time, str):
        return time - datetime.timedelta(minutes=time.minute % cadence)
    else:
        return "not applicable"

def table_1_anova(sinha_df, singh_df):
        flare_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}apj_singh_dataset.csv", parse_dates=["time_start"])
        print(flare_df)
        label = "singh"
        for coincidence in ["all", "coincident", "noncoincident"]:
            if coincidence == "coincident":
                df = flare_df.loc[flare_df["COINCIDENCE"] == True]
            elif coincidence == "noncoincident":
                df = flare_df.loc[flare_df["COINCIDENCE"] == False]
            else:
                df = flare_df
            params = FLARE_PROPERTIES
            # X = flare_df.drop("AR_class", axis=1)
            X = df[params]
            params = X.columns
            y = df["AR_class"]

            f = pd.DataFrame(f_classif(X, y), columns=params).iloc[0]
            f = f.values.reshape(-1, 1)

            f_n = MinMaxScaler().fit_transform(f).ravel()
            f_n = pd.Series(f_n, index=params).sort_values(ascending=False)
            f = pd.Series(f.ravel(), index=params).sort_values(ascending=False)
            f_df = pd.DataFrame({"f_score": f, "f_score_norm": f_n}).rename_axis("parameter")
            f_df.to_csv(f"{other_directory}anova/{label}_anova_{coincidence}.csv")
            print(f_df)


def get_datasets_figure_3(sinha_df, singh_df, dataset_count=20):
    singh_df = singh_df[SINHA_PARAMETERS + ["AR_class", "COINCIDENCE"]]
    for axis_index, coincidence in enumerate(COINCIDENCES):
        occurrences = {
            "KNN": [],
            "RFC": [],
            "LR": [],
            "SVM": [],
        }
        if coincidence == "coincident":
            flares_df = singh_df.loc[singh_df["COINCIDENCE"] == True]
        elif coincidence == "noncoincident":
            flares_df = singh_df.loc[singh_df["COINCIDENCE"] == False]
        else:
            continue
            flares_df = singh_df
        for clf, name, params, param_name in zip(classifiers, names, parameters, param_names):
            if name != "RFC":
                continue
            score = make_scorer(get_tss)
            cv = GridSearchCV(clf, params, cv=10, scoring=score)

            for train_index in range(dataset_count):
                print(f"{name} {train_index + 1}/{dataset_count}")
                train_df, test_df = train_test_split(flares_df, test_size=0.2, stratify=flares_df["AR_class"], random_state=1000 + train_index)
                for col in train_df.columns:
                    if col == "AR_class":
                        continue
                    mean = train_df[col].mean()
                    std = train_df[col].std()
                    test_df[col] = (test_df[col] - mean) / std
                    train_df[col] = (train_df[col] - mean) / std

                X_train, y_train = train_df[SINHA_PARAMETERS], train_df["AR_class"]
                X_test, y_test = test_df[SINHA_PARAMETERS], test_df["AR_class"]
                cv.fit(X_train, y_train)
                if name == "SVM":
                    key1, key2 = param_name
                    occurrences[name].append((cv.best_params_[key1], cv.best_params_[key2]))
                else:
                    occurrences[name].append(cv.best_params_[param_name])

            print(name)
            print(occurrences)
            # with open(f"{other_directory}{coincidence}/sinha_hyperparams.txt",
            #           "w") as fp:
            #     fp.write(json.dumps(occurrences, indent=4))
            print()

def figure_5_classification(sinha_df, singh_df, dataset_count=20):
    # singh_df = singh_df[SINHA_PARAMETERS + ["AR_class", "COINCIDENCE"]]

    # for axis_index, coincidence in enumerate(COINCIDENCES):
    #     if coincidence == "coincident":
    #         # coin
    #         classifiers = [
    #             KNeighborsClassifier(n_neighbors=1),
    #             RandomForestClassifier(n_estimators=120),
    #             LogisticRegression(C=1000),
    #             SVC(C=10, gamma=0.1),
    #         ]
    #         flares_df = singh_df.loc[singh_df["COINCIDENCE"] == True]
    #     elif coincidence == "noncoincident":
    #         # noncoin
    #         classifiers = [
    #             KNeighborsClassifier(n_neighbors=5),
    #             RandomForestClassifier(n_estimators=120),
    #             LogisticRegression(C=100),
    #             SVC(C=1, gamma=1),
    #         ]
    #         flares_df = singh_df.loc[singh_df["COINCIDENCE"] == False]
    #     else:
    #         # all
    #         classifiers = [
    #             KNeighborsClassifier(n_neighbors=1),
    #             RandomForestClassifier(n_estimators=120),
    #             LogisticRegression(C=1),
    #             SVC(C=100, gamma=10),
    #         ]
    flares_df = singh_df
    scores = {
        "KNN": {
            "TSS": [],
            "MAC": [],
            "SSW": [],
            "CSW": []
        },
        "LR": {
            "TSS": [],
            "MAC": [],
            "SSW": [],
            "CSW": []
        },
        "RFC": {
            "TSS": [],
            "MAC": [],
            "SSW": [],
            "CSW": []
        },
        "SVM": {
            "TSS": [],
            "MAC": [],
            "SSW": [],
            "CSW": []
        },
    }
    for train_index in range(dataset_count):
        print(f"{train_index}/{dataset_count}")
        train_df, test_df = train_test_split(flares_df, test_size=0.2, stratify=flares_df["AR_class"])
        for col in SINHA_PARAMETERS:
            if col == "AR_class":
                continue
            mean = train_df[col].mean()
            std = train_df[col].std()
            test_df[col] = (test_df[col] - mean) / std
            train_df[col] = (train_df[col] - mean) / std

        X_train, y_train = train_df[SINHA_PARAMETERS], train_df["AR_class"]
        X_test, y_test = test_df[SINHA_PARAMETERS], test_df["AR_class"]
        for clf, name in zip(classifiers, names):
            clf.fit(X_train, y_train)
            for score_label, score_fn in zip(["TSS", "MAC", "SSW", "CSW"],
                                             [get_tss, get_mac, get_ssw, get_csw]):
                scorer = make_scorer(score_fn)
                s = scorer(clf, X_test, y_test)
                scores[name][score_label].append(s)
                max_scores_dict[name][score_label] = max(max_scores_dict[name][score_label], s)

    import json
    print(max_scores_dict)
    with open(f"{other_directory}/singh_score_new.txt", "w") as fp:
        fp.write(json.dumps(scores, indent=4))

def plot_figure_5():
    d = None

    import json
    with open(f"{other_directory}all/singh_score.txt", "r") as fp:
        d = json.load(fp)
        df = pd.DataFrame(columns=["name", "score", "performance", "error"])

        for name in names:
            for score in ["TSS", "MAC", "SSW", "CSW"]:
                df.loc[df.shape[0]] = [
                    name,
                    score,
                    np.mean(d[name][score]),
                    np.std(d[name][score])
                ]
        df = df.loc[df["score"] == "TSS"].drop("error", axis=1)
        df.to_csv("")
        print(df)
    #
    #
    #         ax = sns.barplot(data=df, x="name", y="performance", hue="score")
    #         plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    #         x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    #         y_coords = [p.get_height() for p in ax.patches]
    #         ax.errorbar(x=x_coords, y=y_coords, yerr=df["error"], fmt="none", c="k")
    #         # ax.set_ylim(bottom=0.8, top=1.0)
    #         ax.set_title(coincidence.capitalize())
    #         plt.tight_layout()
    #         plt.savefig(f"{figure_directory}{coincidence}/singh_classification_performance_with_hyperparams.png")
    #         plt.show()
    #         plt.clf()

    # d = None
    #
    # import json
    # with open(f"{other_directory}sinha_score_loo.txt", "r") as fp:
    #     d = json.load(fp)
    #     df = pd.DataFrame(columns=["name", "score", "performance", "error"])
    #
    #     for name in names:
    #         for score in ["TSS", "MAC", "SSW", "CSW"]:
    #             df.loc[df.shape[0]] = [
    #                 name,
    #                 score,
    #                 np.mean(d[name][score]),
    #                 np.std(d[name][score])
    #             ]
    #     print(df)
    #     df.to_csv(f"{other_directory}sinha_scores.csv")




        # ax = sns.barplot(data=df, x="name", y="performance", hue="score")
        # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        # plt.title(f"")
        # x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
        # y_coords = [p.get_height() for p in ax.patches]
        # ax.errorbar(x=x_coords, y=y_coords, yerr=df["error"], fmt="none", c="k")
        # ax.set_ylim(bottom=0.8, top=1.0)
        # plt.tight_layout()
        # plt.savefig(f"{figure_directory}sinha_classification_performance_preset_random_state.png")
        # plt.show()
        # plt.clf()


def figure_6_plot(sinha_df, singh_df):
    singh_df = singh_df.loc[singh_df["xray_class"] != "N"]
    singh_df = singh_df[SINHA_PARAMETERS + ["AR_class", "COINCIDENCE"]]
    scores = {param: [] for param in SINHA_PARAMETERS}
    scorer = make_scorer(get_tss)
    clf = LogisticRegression(C=1000.0)
    name = "LR"
    for axis_index, coincidence in enumerate(COINCIDENCES):
        if coincidence == "coincident":
            flares_df = singh_df.loc[singh_df["COINCIDENCE"] == True]
        elif coincidence == "noncoincident":
            flares_df = singh_df.loc[singh_df["COINCIDENCE"] == False]
        else:
            flares_df = singh_df
        for param in SINHA_PARAMETERS:
            for train_index in range(20):
                print(f"LR {param} {train_index + 1}/20")
                train_df, test_df = train_test_split(flares_df, test_size=0.2,
                                                     stratify=flares_df["AR_class"],
                                                     random_state=10 + train_index)
                for col in train_df.columns:
                    if col == "AR_class":
                        continue
                    mean = train_df[col].mean()
                    std = train_df[col].std()
                    test_df[col] = (test_df[col] - mean) / std
                    train_df[col] = (train_df[col] - mean) / std

                X_train, y_train = train_df.drop("AR_class", axis=1), train_df[
                    "AR_class"]
                X_test, y_test = test_df.drop("AR_class", axis=1), test_df["AR_class"]

                clf.fit(X_train[param].values.reshape(-1, 1), y_train)
                scores[param].append(scorer(clf, X_test[param].values.reshape(-1, 1), y_test))
        print(scores)


        df = pd.DataFrame(columns=["TSS", "Features"])
        means = [np.mean(scores[param]) for param in SINHA_PARAMETERS]
        for feature, mean in zip(SINHA_PARAMETERS, means):
            df.loc[df.shape[0]] = [mean, feature]
        print(df)
        ax = sns.barplot(df, y="Features", x="TSS")

        for p in ax.patches:
            ax.annotate("%.4f" % p.get_width(),
                        xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords='offset points', ha="left",
                        va="center")
        plt.title(coincidence.capitalize())
        plt.tight_layout()
        plt.savefig(f"{figure_directory}{coincidence}/singh_logistic_regression_no_n.png")
        plt.show()

def figure_10_plot(sinha_df, singh_df):
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (1 / 6., 0.0, 0.0),
                     (1 / 2., 0.8, 1.0),
                     (5 / 6., 1.0, 1.0),
                     (1.0, 0.4, 1.0)),

             'green': ((0.0, 0.0, 0.4),
                       (1 / 6., 1.0, 1.0),
                       (1 / 2., 1.0, 0.8),
                       (5 / 6., 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

             'blue': ((0.0, 0.0, 0.0),
                      (1 / 6., 0.0, 0.0),
                      (1 / 2., 0.9, 0.9),
                      (5 / 6., 0.0, 0.0),
                      (1.0, 0.0, 0.0))

             }

    cmap = "PiYG"
    param_order = ["TOTUSJH", "USFLUX", "TOTUSJZ",
                   "R_VALUE", "TOTABSTWIST", "TOTBSQ",
                   "TOTPOT", "AREA_ACR", "SAVNCPP",
                   "TOTFZ", "ABSNJZH", "MEANPOT",
                   "SHRGT45", "EPSZ"]
    print(singh_df)
    coin_df = singh_df.loc[singh_df["COINCIDENCE"] == True]
    noncoin_df = singh_df.loc[singh_df["COINCIDENCE"] == False]
    plt.figure(figsize=(10, 9), dpi=100)
    g = sns.heatmap(coin_df[SINHA_PARAMETERS].corr(), cmap=cmap, vmin=-1.0, vmax=1.0,
                    fmt=".2f", annot=True)
    plt.tight_layout()
    plt.savefig("singh_correlation_coincident_plot.png")
    plt.show()

    plt.figure(figsize=(10, 9), dpi=100)
    g = sns.heatmap(noncoin_df[SINHA_PARAMETERS].corr(), cmap=cmap, vmin=-1.0, vmax=1.0,
                    fmt=".2f", annot=True)
    plt.tight_layout()
    plt.savefig("singh_correlation_noncoincident_plot.png")
    plt.show()

    plt.figure(figsize=(10, 9), dpi=100)
    g = sns.heatmap(singh_df[SINHA_PARAMETERS].corr(), cmap=cmap, vmin=-1.0, vmax=1.0,
                    fmt=".2f", annot=True)
    plt.tight_layout()
    plt.savefig("singh_correlation_plot.png")
    plt.show()

    # plt.figure(figsize=(10, 9), dpi=100)
    # g = sns.heatmap(sinha_df[SINHA_PARAMETERS].corr(), cmap=cmap, vmin=-1.0, vmax=1.0,
    #                 fmt=".2f", annot=True)
    # plt.tight_layout()
    # plt.show()
    #


def hyperparam_plot():
    knn_values = [i for i in range(1, 17)]
    knn_counts = [4, 0, 7, 0, 3, 0, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0]

    plt.bar(knn_values, knn_counts)
    plt.xticks(knn_values)
    plt.xlabel("K")
    plt.ylabel("Occurrences")
    plt.tight_layout()
    plt.savefig("knn_all_hyperparameters.png")
    plt.show()

    lr_counts = [0, 0, 0, 0, 3, 4, 7, 6, 0]
    lr_values = [i for i in range(len(lr_counts))]


    plt.bar(lr_values, lr_counts)
    plt.xlabel("C")
    plt.ylabel("Occurrences")
    plt.xticks(range(len(lr_counts)), labels=[10**e for e in range(-4, 5)], rotation=45)
    plt.yticks([i for i in range(max(lr_counts) + 1)])
    plt.tight_layout()
    plt.savefig("lr_all_hyperparameters.png")
    plt.show()

    rfc_counts = [4, 7, 3, 1, 1, 0, 1, 0, 1, 2]
    rfc_values = [10, 120, 230, 340, 450, 560, 670, 780, 890, 1000]

    plt.bar(range(len(rfc_counts)), rfc_counts)
    plt.xlabel("Number of decision trees")
    plt.ylabel("Occurrences")
    plt.xticks(range(len(rfc_counts)), labels=rfc_values,
               rotation=45)
    plt.yticks([i for i in range(max(rfc_counts) + 1)])
    plt.tight_layout()
    plt.savefig("rfc_all_hyperparameters.png")
    plt.show()


    data = np.zeros((6, 6))
    data[1, 3] = 8
    data[0, 4] = 5
    data[1, 4] = 3
    data[0, 2] = 1
    data[2, 3] = 2
    data[0, 5] = 1
    yticks = [10**e for e in range(-3, 3)]
    yticks.reverse()
    ax = sns.heatmap(data, annot=True,
                xticklabels=[10**e for e in range(-4, 2)],
                yticklabels=yticks,
                     cmap="RdYlGn",
                     cbar=False)
    ax.set(xlabel="gamma", ylabel="C")
    plt.tight_layout()
    plt.savefig("svm_all_hyperparameters.png")
    plt.show()

def best_features_for_clf():
    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        LogisticRegression(C=100),
        RandomForestClassifier(n_estimators=120),
        SVC(C=10, gamma=0.1)
    ]
    names = [
        "KNN",
        "LR",
        "RFC",
        "SVM"
    ]
    score_names = [
        "tss",
        "mac",
        "ssw",
        "csw"
    ]
    score_functions = [
        get_tss,
        get_mac,
        get_ssw,
        get_csw
    ]

    singh_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}singh_nbmx_data.csv",
                           index_col="index")
    classifier_df = pd.DataFrame(columns=["name", "parameter"] + score_names)
    for parameter in FLARE_PROPERTIES:
        X = singh_df[[parameter]]
        y = singh_df["AR_class"].values
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        for name, clf in zip(names, classifiers):
            y_true, y_pred = [], []
            i = 0
            for train_index, test_index in loo.split(X):
                i += 1
                print(f"{name} {clf} {parameter} {i}/{len(X)}")
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred.append(clf.predict(X_test))
                y_true.append(y_test)
            tss = get_tss(y_true, y_pred)
            mac = get_mac(y_true, y_pred)
            ssw = get_ssw(y_true, y_pred)
            csw = get_csw(y_true, y_pred)
            classifier_df.loc[classifier_df.shape[0]] = [
                name,
                parameter,
                tss,
                mac,
                ssw,
                csw
            ]
            classifier_df.to_csv(f"{metrics_directory}individual_feature_performances_by_classifier.csv")

def main() -> None:
    sinha_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_dataset.csv")
    # sinha_df.index.names = ["index"]
    # singh_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}singh_nbmx_data.csv", index_col="index")
    singh_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}apj_singh_dataset.csv", index_col=0, parse_dates=["time_start"])
    # table_1_anova(sinha_df, singh_df)
    # figure_5_classification(sinha_df, singh_df)
    # table_1_anova(sinha_df, singh_df)
    # best_features_for_clf()

    figure_10_plot(sinha_df, singh_df)
    exit(1)
    hyperparam_plot()
    # print(singh_df)
    # plot_figure_5()
    # hyperparam_plot()
    # figure_10_plot(sinha_df, singh_df)
    # table_1_anova(sinha_df, singh_df)
    # get_datasets_figure_3(sinha_df, singh_df)
    # exit(1)
    # get_datasets_figure_3(sinha_df, singh_df)
    # figure_5_classification(sinha_df, singh_df)
    # figure_5_classification(sinha_df, singh_df)
    # plot_figure_5()
    # print(max_scores_dict)
    # print(random_states_dict)
    # print(parameters)
    # figure_6_plot(sinha_df, singh_df)

if __name__ == "__main__":
    main()
