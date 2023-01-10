################################################################################
# Filename: sinha_study.py
# Description: Todo
################################################################################

# Custom Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import json

from source.utilities import *

# Disable Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Experiment Name (No Acronyms)
experiment = "sinha_study_lr_svm"
experiment_caption = experiment.title().replace("_", " ")

# SINHA_PARAMETERS = [
#     "TOTUSJH",
#     "USFLUX",
#     "TOTUSJZ",
#     "R_VALUE",
#     "TOTPOT",
#     "AREA_ACR",
#     "SAVNCPP",
#     "ABSNJZH",
#     "MEANPOT",
#     "SHRGT45",
# ]

SINHA_PARAMETERS = [
    "TOTUSJH",
    "TOTPOT",
    "TOTUSJZ",
    "ABSNJZH",
    "SAVNCPP",
    "USFLUX",
    "AREA_ACR",
    "MEANPOT",
    "R_VALUE",
    "SHRGT45",
    "EPSZ",
    "TOTBSQ",
    "TOTFZ",
    "TOTABSTWIST"
]

score_names = [
    "TSS",
    "MAC",
    "SSW",
    "CSW",
]

names = [
# "LR",
"SVM",
]

classifiers = [
# LogisticRegression(),
SVC(),
]

lr_params = {
    "solver": ["lbfgs", "newton-cg", "liblinear", "saga"],
    "C": [10**e for e in range(-5, 5)],
    "tol": [10**e for e in range(-5, -2)],
    "dual": [True, False],
}

svm_params = {
    "kernel": ["sigmoid"],
    "C": [10**e for e in range(-1, 3)],
    "gamma": [10**e for e in range(-3, 2)],
    "tol": [10**e for e in range(-5, -2)],
    "coef0": [-1, 0, 1]
}


max_scores_dict = {name: {score: 0 for score in score_names} for name in names}
random_states_dict = {name: {score: 0 for score in score_names} for name in names}

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


def figure_5_classification(sinha_df, dataset_count=20):
    flares_df = sinha_df
    # with open(f"{other_directory}lr_sinha_scores.csv", "w") as fp:
    #     fp.write("TSS_mean,TSS_std,SSW_mean,SSW_std,solver,penalty,C,dual,l1_ratio,tol,\n")
    # with open(f"{other_directory}svm_sinha_scores.csv", "w") as fp:
    #     fp.write("TSS_mean,TSS_std,SSW_mean,SSW_std,kernel,C,gamma,degree,coef0,tol,\n")


    def train_test_clf(classifier, classifier_name, params):
        tss_scores = []
        ssw_scores = []
        print(params)
        print(classifier.get_params().keys())
        classifier.set_params(**params)
        print(f"{classifier_name} : {params}")
        for train_index in range(dataset_count):
            print(f"{train_index}/{dataset_count}")
            train_df, test_df = train_test_split(flares_df, test_size=0.2,
                                                 stratify=flares_df["AR_class"],
                                                 random_state=100 + train_index)
            for col in train_df.columns:
                if col == "AR_class":
                    continue
                mean = train_df[col].mean()
                std = train_df[col].std()
                test_df[col] = (test_df[col] - mean) / std
                train_df[col] = (train_df[col] - mean) / std
            X_train, y_train = train_df[SINHA_PARAMETERS], train_df["AR_class"]
            X_test, y_test = test_df[SINHA_PARAMETERS], test_df["AR_class"]
            classifier.fit(X_train, y_train)
            tss_scorer = make_scorer(get_tss)
            ssw_scorer = make_scorer(get_tss)
            tss_scores.append(tss_scorer(classifier, X_test, y_test))
            ssw_scores.append(ssw_scorer(classifier, X_test, y_test))

        with open(f"{other_directory}{classifier_name.lower()}_sinha_scores.csv", "a") as fp:
            if classifier_name in "LR":
                fp.write(f"{np.mean(tss_scores)},"
                         f"{np.std(tss_scores)},"
                         f"{np.mean(ssw_scores)},"
                         f"{np.std(ssw_scores)},"
                         f"{params['solver']},"
                         f"{params['penalty']},"
                         f"{params['C']},"
                         f"{params['dual']},"
                         f"{params['l1_ratio']},"
                         f"{params['tol']},\n")
            else:
                fp.write(f"{np.mean(tss_scores)},"
                         f"{np.std(tss_scores)},"
                         f"{np.mean(ssw_scores)},"
                         f"{np.std(ssw_scores)},"
                         f"{params['kernel']},"
                         f"{params['C']},"
                         f"{params['gamma']},"
                         f"{params['degree']},"
                         f"{params['coef0']},"
                         f"{params['tol']},\n")



    for clf, name in zip(classifiers, names):
        if name in "LR":
            for solver in lr_params["solver"]:
                for C in lr_params["C"]:
                    for tol in lr_params["tol"]:
                        for toggle in [True, False]:
                            if solver in "liblinear" and toggle is False:
                                continue
                            if solver in "saga" and toggle is False:
                                continue
                            penalty = "elasticnet" if solver in "saga" and toggle is True else "l2"
                            d = {
                                "solver": solver,
                                "penalty": "elasticnet" if solver in "saga" and toggle is True
                                else "l2",
                                "C": C,
                                "tol": tol,
                                "dual": True if solver in "liblinear" and toggle is True
                                else False,
                                "l1_ratio": 0.5 if penalty in "elasticnet" else None
                            }
                            train_test_clf(clf, name, d)

        elif name in "SVM":
            degree = 3
            for kernel in svm_params["kernel"]:
                for C in svm_params["C"]:
                    for tol in svm_params["tol"]:
                        for gamma in svm_params["gamma"]:
                            for coef0 in svm_params["coef0"]:
                                d = {"kernel": kernel,
                                     "C": C,
                                     "gamma": gamma,
                                     "degree": degree,
                                     "coef0": coef0,
                                     "tol": tol,
                                     }
                                clf.set_params(**d)
                                train_test_clf(clf, name, d)




def plot_figure_5():
    d = None

    import json
    with open(f"{other_directory}sinha_score_loo.txt", "r") as fp:
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
        print(df)


        ax = sns.barplot(data=df, x="name", y="performance", hue="score")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.title(f"Leave One Out Testing")
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]
        ax.errorbar(x=x_coords, y=y_coords, yerr=df["error"], fmt="none", c="k")
        ax.set_ylim(bottom=0.8, top=1.0)
        plt.tight_layout()
        plt.savefig(f"{figure_directory}sinha_classification_performance_loo.png")
        plt.show()
        plt.clf()



def main() -> None:
    sinha_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_dataset.csv")
    sinha_df.index.names = ["index"]
    singh_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}singh_nbmx_data.csv", index_col="index")
    figure_5_classification(sinha_df)

if __name__ == "__main__":
    main()
