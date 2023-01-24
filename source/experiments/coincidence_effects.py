################################################################################
# Filename: coincidence_effects.py
# Description:
################################################################################

# Custom Imports
import pandas as pd
from sklearn.linear_model import LogisticRegression

from source.utilities import *

# df = pd.concat([pd.read_csv(f"{FLARE_LIST_DIRECTORY}0h_24h/b_list.txt",
#                                 header=0, index_col="index"),
#                     pd.read_csv(f"{FLARE_LIST_DIRECTORY}0h_24h/mx_list.txt",
#                                 header=0, index_col="index")
#                     ])
df = pd.concat([pd.read_csv(f"{FLARE_DATA_DIRECTORY}time_series_means/original_coincidence_definition/0h_24h/b_0h_24h_mean_dataset.csv",
                                header=0),
                    pd.read_csv(f"{FLARE_DATA_DIRECTORY}time_series_means/original_coincidence_definition/0h_24h/mx_0h_24h_mean_dataset.csv",
                                header=0)
                    ])
df = df.dropna()
df["AR_class"] = (df["xray_class"].str.contains("M") | df["xray_class"].str.contains("X")).astype(int)

training_df = df.loc[(df["time_start"].str.contains("2012")) |
                              (df["time_start"].str.contains("2013"))
                              | (df["time_start"].str.contains("2014"))
                              | (df["time_start"].str.contains("2015"))]
testing_df = df.loc[(df["time_start"].str.contains("2010")) |
                              (df["time_start"].str.contains("2011"))
                              | (df["time_start"].str.contains("2016"))
                              | (df["time_start"].str.contains("2017"))
                              | (df["time_start"].str.contains("2018"))
                              | (df["time_start"].str.contains("2019"))
                              | (df["time_start"].str.contains("2020"))]
df['time_start'] = pd.to_datetime(df['time_start'],
                                  format='%Y-%m-%d %H:%M:%S')
training_df['time_start'] = pd.to_datetime(training_df['time_start'],
                                  format='%Y-%m-%d %H:%M:%S')
testing_df['time_start'] = pd.to_datetime(testing_df['time_start'],
                                  format='%Y-%m-%d %H:%M:%S')
df = df.sort_values(by="time_start")
df.reset_index(inplace=True)
training_df = training_df.sort_values(by="time_start")
testing_df = testing_df.sort_values(by="time_start")

coincidence_df = df.loc[df["COINCIDENCE"] == True]
noncoincidence_df = df.loc[df["COINCIDENCE"] == False]

first_coin_df = coincidence_df.groupby('nar').head(1)

params = [
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
        "LDA",
        "KNN",
        "LR",
        "RFC",
        "SVM",
]

classifiers = [
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(3),
    LogisticRegression(C=1000),
    RandomForestClassifier(n_estimators=120),
    SVC(gamma=0.001, C=100),
]

def get_tss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # tn, fp, fn, tp = cm.ravel()
    tp, fn, fp, tn = cm.ravel()  # Makes top left TP and bottom right TN
    detection_rate = tp / float(tp + fn)
    false_alarm_rate = fp / float(fp + tn)
    tss = detection_rate - false_alarm_rate
    return tss


def get_mac(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # tn, fp, fn, tp = cm.ravel()
    tp, fn, fp, tn = cm.ravel()  # Makes top left TP and bottom right TN
    detection_rate = tp / float(tp + fn)
    true_negative_rate = tn / float(fp + tn)
    mac = 0.5 * (detection_rate + true_negative_rate)
    return mac


def get_ssw(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # tn, fp, fn, tp = cm.ravel()
    tp, fn, fp, tn = cm.ravel()  # Makes top left TP and bottom right TN
    ssw = (tp - fn) / (tp + fn)
    return ssw


def get_csw(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # tn, fp, fn, tp = cm.ravel()
    tp, fn, fp, tn = cm.ravel()  # Makes top left TP and bottom right TN
    csw = (tn - fp) / (tn + fp)
    return csw


score_names = [
    "TSS",
    "MAC",
    "SSW",
    "CSW",
]

score_functions = [
    get_tss,
    get_mac,
    get_ssw,
    get_csw
]

# Experiment Name (No Acronyms)
experiment = "coincidence_effects"
experiment_caption = experiment.title().replace("_", " ")

# ------------------------------------------------------------------------
# Place any results in the directory for the current experiment.
now_string, figure_directory, metrics_directory, other_directory = \
    build_experiment_directories(experiment)


def ar_counts():
    d = {}
    for ar, count in coincidence_df["nar"].value_counts(ascending=True).iteritems():
        if count not in d:
            d[count] = 1
        else:
            d[count] += 1
    print(d)
    print(coincidence_df["nar"].value_counts(ascending=True))


def ar_loo():
    with open(f"{metrics_directory}first_coincident_bmx_flare_loo_classification.csv", "a") as fp:
        fp.write("classifier,TSS,MAC,SSW,CSW,\n")

    for clf, name in zip(classifiers, names):
        X = first_coin_df[params]
        y = first_coin_df["AR_class"].values
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        y_true, y_pred = [], []
        for index, (train_index, test_index) in enumerate(loo.split(X)):
            print(f"{name} {index}/{len(X)}")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for col in params:
                if col == "AR_class":
                    continue
                mean = X_train[col].mean()
                std = X_train[col].std()
                X_test[col] = (X_test[col] - mean) / std
                X_train[col] = (X_train[col] - mean) / std

            clf.fit(X_train, y_train)
            y_pred.append(clf.predict(X_test)[0])
            y_true.append(y_test[0])

        with open(f"{metrics_directory}first_coincident_bmx_flare_loo_classification.csv", "a") as fp:
            fp.write(f"{name},")
            for score, score_fn in zip(score_names, score_functions):
                fp.write(f"{score_fn(y_true, y_pred)},")
            fp.write("\n")


def train_noncoin_test_first_coin():
    with open(f"{metrics_directory}train_noncoincident_test_first_coincident_bmx_flare_classification.csv", "a") as fp:
        fp.write("classifier,TSS,MAC,SSW,CSW,\n")

    train_df = noncoincidence_df
    test_df = first_coin_df
    print(test_df.loc[test_df["xray_class"] == "B"])

    X_train, X_test = train_df[params], test_df[params]
    y_train, y_test = train_df["AR_class"].values, test_df["AR_class"].values

    for col in params:
        if col == "AR_class":
            continue
        mean = X_train[col].mean()
        std = X_train[col].std()
        X_test[col] = (X_test[col] - mean) / std
        X_train[col] = (X_train[col] - mean) / std

    for clf, name in zip(classifiers, names):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_true = y_test

        with open(f"{metrics_directory}train_noncoincident_test_first_coincident_bmx_flare_classification.csv", "a") as fp:
            fp.write(f"{name},")
            for score, score_fn in zip(score_names, score_functions):
                fp.write(f"{score_fn(y_true, y_pred)},")
            fp.write("\n")


def peak_years_testing():
    for coincidence in ["all", "coincident", "noncoincident"]:
        with open(f"{metrics_directory}training_on_end_bmx_flare_loo_classification_{coincidence}.csv", "a") as fp:
            fp.write("classifier,TSS,MAC,SSW,CSW,\n")

        for clf, name in zip(classifiers, names):
            if coincidence == "coincident":
                X_train, X_test = training_df.loc[training_df["COINCIDENCE"] == True][params],\
                                  testing_df.loc[testing_df["COINCIDENCE"] == True][params]
                y_train, y_test = training_df.loc[training_df["COINCIDENCE"] == True]["AR_class"].values,\
                                  testing_df.loc[testing_df["COINCIDENCE"] == True]["AR_class"].values
            elif coincidence == "noncoincident":
                X_train, X_test = \
                training_df.loc[training_df["COINCIDENCE"] == False][params], \
                testing_df.loc[testing_df["COINCIDENCE"] == False][params]
                y_train, y_test = \
                training_df.loc[training_df["COINCIDENCE"] == False][
                    "AR_class"].values, \
                testing_df.loc[testing_df["COINCIDENCE"] == False][
                    "AR_class"].values
            else:
                X_train, X_test = training_df[params], testing_df[params]
                y_train, y_test = training_df["AR_class"].values, testing_df["AR_class"].values
            for col in params:
                if col == "AR_class":
                    continue
                mean = X_train[col].mean()
                std = X_train[col].std()
                X_test[col] = (X_test[col] - mean) / std
                X_train[col] = (X_train[col] - mean) / std

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_true = y_test

            with open(f"{metrics_directory}training_on_end_bmx_flare_loo_classification_{coincidence}.csv", "a") as fp:
                fp.write(f"{name},")
                print(y_true, y_pred)
                for score, score_fn in zip(score_names, score_functions):
                    fp.write(f"{score_fn(y_true, y_pred)},")
                fp.write("\n")

            with open(f"{metrics_directory}training_on_end_bmx_flare_loo_classification_{coincidence}.csv", "a") as fp:
                fp.write(f"{name},")
                print(y_true, y_pred)
                for score, score_fn in zip(score_names, score_functions):
                    fp.write(f"{score_fn(y_true, y_pred)},")
                fp.write("\n")


def first_coin_flare_and_other_ars():
    with open(f"{metrics_directory}first_coincident_in_one_ar_and_other_coincident_in_other_ars_bmx_flare_loo_classification.csv", "w") as fp:
        fp.write("classifier,TSS,MAC,SSW,CSW,\n")

    for clf, name in zip(classifiers, names):
        y_true, y_pred = [], []
        for index, row in first_coin_df.iterrows():
            print(f"{name} {index}/{len(first_coin_df)}")
            time_start = row["time_start"]
            nar = row["nar"]
            train_df = coincidence_df.loc[coincidence_df["nar"] != nar]
            test_df = coincidence_df.loc[(coincidence_df["nar"] == nar) & (coincidence_df["time_start"] != time_start)]
            if test_df.empty:
                continue


            train_X = train_df[params]
            train_y = train_df["AR_class"].values
            test_X = test_df[params]
            test_y = test_df["AR_class"].values

            for col in params:
                mean = train_X[col].mean()
                std = train_X[col].std()
                test_X[col] = (test_X[col] - mean) / std
                train_X[col] = (train_X[col] - mean) / std

            clf.fit(train_X, train_y)
            y_pred += list(clf.predict(test_X))
            y_true += list(test_y)

            with open(f"{other_directory}{name}_first_coincident_in_one_ar_and_other_coincident_in_other_ars_bmx_flare_loo_classification_confusion_matrix.txt", "w") as fp:
                fp.write(f"{name} Confusion Matrix\n")
                fp.write(str(confusion_matrix(y_true, y_pred)) + "\n")
                fp.write(str(classification_report(y_true, y_pred)))

        with open(f"{metrics_directory}first_coincident_in_one_ar_and_other_coincident_in_other_ars_bmx_flare_loo_classification.csv", "a") as fp:
            fp.write(f"{name},")
            for score, score_fn in zip(score_names, score_functions):
                fp.write(f"{score_fn(y_true, y_pred)},")
            fp.write("\n")


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)


    # ar_counts()
    # ar_loo()
    # train_noncoin_test_first_coin()
    # train_noncoin_test_first_coin()

    # second_coin_df = df.groupby('nar').head(2)
    # index = coincidence_df["nar"].value_counts().reset_index(name="count").query("count >= 2")["index"]
    # print(index)

    # print(coincidence_df.groupby("nar").head(1).shape[0])

    first_coin_flare_and_other_ars()

    # print()
    # print(df.groupby('nar').head(2))
    # peak_years_testing()

    # classes = ["B", "M", "X"]
    # for coin in [True, False]:
    #     df = training_df.loc[training_df["COINCIDENCE"] == coin]
    #     df2 = testing_df.loc[testing_df["COINCIDENCE"] == coin]
    #     for c in classes:
    #         train_df = df.loc[df["xray_class"] == c]
    #         print(train_df.shape[0], end=" ")
    #     print()
    #     for c in classes:
    #         test_df = df2.loc[df2["xray_class"] == c]
    #         print(test_df.shape[0], end=" ")
    #     print()
    #     print()
    # for c in classes:
    #     train_df = training_df.loc[training_df["xray_class"] == c]
    #     print(train_df.shape[0], end=" ")
    # print()
    # for c in classes:
    #     test_df = testing_df.loc[testing_df["xray_class"] == c]
    #     print(test_df.shape[0], end=" ")


if __name__ == "__main__":
    main()
