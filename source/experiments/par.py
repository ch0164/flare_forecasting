################################################################################
# Filename: sinha_study.py
# Description: Todo
################################################################################

# Custom Imports
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

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

names = [
        "KNN",
        "LR",
        "RFC",
        "SVM",
]

classifiers = [
    KNeighborsClassifier(),
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(),
]

parameters = [
    dict(n_neighbors=list(range(1, 16 + 1))),
    dict(C=[10**e for e in range(-5, 5)]),
    dict(n_estimators=[10 + 110 * i for i in range(10)]),
    dict(C=[10**e for e in range(-4, 3)],
         gamma=[10**e for e in range(-5, 2)]),
]

param_names = [
    "n_neighbors",
    "C",
    "n_estimators",
    ("C", "gamma")
]

occurrences = {
    "KNN": {p: 0 for p in parameters[0]["n_neighbors"]},
    "LR": {p: 0 for p in parameters[1]["C"]},
    "RFC": {p: 0 for p in parameters[2]["n_estimators"]},
    "SVM": {
        "C": {p: 0 for p in parameters[3]["C"]},
        "gamma": {p: 0 for p in parameters[3]["gamma"]} },
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
    detection_rate = tp / float(tp + fn)
    true_negative_rate = tn / float(fp + tn)
    ssw = (tp - fn) / (tp + fn)
    return ssw


def get_csw(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    csw = (tn - fp) / (tn + fp)
    return csw


def table_1_anova(sinha_df, singh_df):
        flare_df = singh_df
        label = "singh"
        flare_df = flare_df.loc[flare_df["xray_class"] != "N"]
        params = SINHA_PARAMETERS
        # X = flare_df.drop("AR_class", axis=1)
        X = flare_df[params]
        params = X.columns
        y = flare_df["AR_class"]

        f = pd.DataFrame(f_classif(X, y), columns=params).iloc[0]
        f = f.values.reshape(-1, 1)

        f_n = MinMaxScaler().fit_transform(f).ravel()
        f_n = pd.Series(f_n, index=params).sort_values(ascending=False)
        f = pd.Series(f.ravel(), index=params).sort_values(ascending=False)
        f_df = pd.DataFrame({"f_score": f, "f_score_norm": f_n}).rename_axis("parameter")
        f_df.to_csv(f"{other_directory}anova/{label}_anova_no_n.csv")
        print(f_df)


def get_datasets_figure_3(sinha_df, singh_df):
    singh_df = singh_df[SINHA_PARAMETERS + ["AR_class"]]
    for clf, name, params, param_name in zip(classifiers, names, parameters, param_names):
        if name == "RFC":
            continue
        score = make_scorer(get_tss)
        cv = GridSearchCV(clf, params, cv=10, scoring=score)

        for train_index in range(20):
            print(f"{name} {train_index + 1}/20")
            train_df, test_df = train_test_split(sinha_df, test_size=0.2, stratify=sinha_df["AR_class"])
            for col in train_df.columns:
                if col == "AR_class":
                    continue
                mean = train_df[col].mean()
                std = train_df[col].std()
                test_df[col] = (test_df[col] - mean) / std
                train_df[col] = (train_df[col] - mean) / std

            X_train, y_train = train_df.drop("AR_class", axis=1), train_df["AR_class"]
            X_test, y_test = test_df.drop("AR_class", axis=1), test_df["AR_class"]
            cv.fit(X_train, y_train)
            if name == "SVM":
                key1, key2 = param_name
                occurrences[name][key1][cv.best_params_[key1]] += 1
                occurrences[name][key2][cv.best_params_[key2]] += 1
            else:
                occurrences[name][cv.best_params_[param_name]] += 1

        print(name)
        print(occurrences)
        print()

def figure_5_classification(sinha_df, singh_df):
    singh_df = singh_df[SINHA_PARAMETERS + ["AR_class"]]
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
    for train_index in range(20):
        print(f"{train_index}/20")
        train_df, test_df = train_test_split(singh_df, test_size=0.2, stratify=singh_df["AR_class"])
        for col in train_df.columns:
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
                scores[name][score_label].append(scorer(clf, X_test, y_test))

    import json
    with open(f"{other_directory}singh_score.txt", "w") as fp:
        fp.write(json.dumps(scores, indent=4))

def plot_figure_5():
    d = None
    import json
    with open(f"{other_directory}sinha_score.txt", "r") as fp:
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


    ax = sns.barplot(data=df, x="name", y="performance", hue="score")
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=df["error"], fmt="none", c="k")
    plt.ylim(bottom=0.8, top=1.0)
    plt.savefig(f"{figure_directory}sinha_classification_performance.png")
    plt.show()

def main() -> None:
    sinha_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_dataset.csv")
    sinha_df.index.names = ["index"]
    singh_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}singh_nbmx_data.csv", index_col="index")
    # table_1_anova(sinha_df, singh_df)
    # get_datasets_figure_3(sinha_df, singh_df)
    # figure_5_classification(sinha_df, singh_df)
    plot_figure_5()
    # print(parameters)

if __name__ == "__main__":
    main()


