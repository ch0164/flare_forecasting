################################################################################
# Filename: experiment_template.py
# Description: This file is a correlation for easy creation of new experiments.
################################################################################

# Custom Imports
from sklearn.linear_model import LogisticRegression

from source.utilities import *

# Disable Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

experiment = "swan_sf_mvts"
now_string, figure_directory, metrics_directory, other_directory = \
    build_experiment_directories(experiment)


partition1_dir = r"C:\Users\youar\Desktop\swan-sf\partition1"
partition2_dir = r"C:\Users\youar\Desktop\swan-sf\partition2"


class MVTSSample:
    def __init__(self, flare_type: str, start_time: datetime,
                 end_time: datetime, data: pd.DataFrame):
        self._flare_type = flare_type
        self._start_time = start_time
        self._end_time = end_time
        self._data = data

    def get_flare_type(self):
        return self._flare_type

    def get_start_time(self):
        return self._start_time

    def get_end_time(self):
        return self._end_time

    def get_data(self):
        return self._data


def read_mvts_instance(data_dir: str,
                       file_name: str) -> MVTSSample:  # Finished!
    # Get flare type from file name
    flare_type = file_name[0:2]
    start_time, end_time, data = None, None, None

    try:
        # Get start time from file name
        start = file_name.find('s2')
        start_time = file_name[start + 1: start + 20]
        start_time = start_time.replace("T", " ")
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H_%M_%S")

        # Get end time from file name
        end = file_name.find('e2')
        end_time = file_name[end + 1: end + 20]
        end_time = end_time.replace("T", " ")
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H_%M_%S")
    except ValueError:
        print(ValueError)
        pass

    # Get data from csv file
    try:
        data = pd.read_csv(data_dir + "/" + file_name, sep="\t")
    except ValueError:
        print(ValueError)
        pass

    # Make mvts object
    mvts = MVTSSample(flare_type, start_time, end_time, data)
    return mvts


def calculate_descriptive_features(data: pd.DataFrame) -> pd.DataFrame:  # Finished!
    variates_to_calc_on = ['R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ',
                           'ABSNJZH', 'SAVNCPP',
                           'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR',
                           'SHRGT45', 'MEANGAM', 'MEANGBT',
                           'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
                           'MEANALP', 'TOTFX']
    features_to_return = ['R_VALUE_MEDIAN', 'R_VALUE_STDDEV',
                          'TOTUSJH_MEDIAN', 'TOTUSJH_STDDEV',
                          'TOTBSQ_MEDIAN', 'TOTBSQ_STDDEV',
                          'TOTPOT_MEDIAN', 'TOTPOT_STDDEV',
                          'TOTUSJZ_MEDIAN', 'TOTUSJZ_STDDEV',
                          'ABSNJZH_MEDIAN', 'ABSNJZH_STDDEV',
                          'SAVNCPP_MEDIAN', 'SAVNCPP_STDDEV',
                          'USFLUX_MEDIAN', 'USFLUX_STDDEV',
                          'TOTFZ_MEDIAN', 'TOTFZ_STDDEV',
                          'MEANPOT_MEDIAN', 'MEANPOT_STDDEV',
                          'EPSZ_MEDIAN', 'EPSZ_STDDEV',
                          'MEANSHR_MEDIAN', 'MEANSHR_STDDEV',
                          'SHRGT45_MEDIAN', 'SHRGT45_STDDEV',
                          'MEANGAM_MEDIAN', 'MEANGAM_STDDEV',
                          'MEANGBT_MEDIAN', 'MEANGBT_STDDEV',
                          'MEANGBZ_MEDIAN', 'MEANGBZ_STDDEV',
                          'MEANGBH_MEDIAN', 'MEANGBH_STDDEV',
                          'MEANJZH_MEDIAN', 'MEANJZH_STDDEV',
                          'TOTFY_MEDIAN', 'TOTFY_STDDEV',
                          'MEANJZD_MEDIAN', 'MEANJZD_STDDEV',
                          'MEANALP_MEDIAN', 'MEANALP_STDDEV',
                          'TOTFX_MEDIAN', 'TOTFX_STDDEV']
    # Create empty data frame for return with named columns
    df = pd.DataFrame(columns=features_to_return)

    # For each element append to temp list
    list2add = []
    for d in variates_to_calc_on:
        l = data[d].to_numpy()
        median = np.median(l)
        std = np.std(l)
        list2add.append(median)
        list2add.append(std)
        continue

    df.loc[len(df)] = list2add
    return df


def process_partition(partition_location: str, filename: str):  # NEEDS WORK!
    abt_header = ['FLARE_TYPE'] + [f'R_VALUE_{i}' for i in range(1, 61)] + [f'TOTUSJZ_{i}' for i in range(1, 61)] + [f'TOTUSJH_{i}' for i in range(1, 61)]
    abt = pd.DataFrame(columns=abt_header)

    # Get lists of data from partition
    FL = os.listdir(partition_location + "/FL")
    NF = os.listdir(partition_location + "/NF")

    count = 0
    for d in FL + NF:
        if d in FL:
            mvs = read_mvts_instance(partition_location + '/FL', d)
        else:
            mvs = read_mvts_instance(partition_location + '/NF', d)
        r_values = mvs.get_data()["R_VALUE"].tolist()
        totus_jz_values = mvs.get_data()["TOTUSJZ"].tolist()
        totus_jh_values = mvs.get_data()["TOTUSJH"].tolist()
        abt.loc[len(abt)] = [mvs.get_flare_type()] + r_values + totus_jz_values + totus_jh_values

        ''' Limit to 10000 files for testing'''
        count +=1
        print(f"{count}: {d}")
        if count >= 10000:
            break
        continue

    abt.to_csv(filename)
    return abt


def calc_tss(y_true=None, y_predict=None):
    """
    Calculates the true skill score for binary classification based on the output of the confusion
    table function
    """
    scores = confusion_matrix(y_true, y_predict).ravel()
    TN, FP, FN, TP = scores
    print('TN={0}\tFP={1}\tFN={2}\tTP={3}'.format(TN, FP, FN, TP))
    tp_rate = TP / float(TP + FN) if TP > 0 else 0
    fp_rate = FP / float(FP + TN) if FP > 0 else 0

    return tp_rate - fp_rate


def get_ar_class(flare_class: str) -> int:
    if "M" in flare_class or "X" in flare_class:
        return 1
    else:
        return 0


from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def main() -> None:
    # df = process_partition(partition1_dir,
    #                        f"{other_directory}parition2_r_values_totusjz_totusjh.csv")
    #
    # # Replace nan with mean of that row
    # m = df.mean(axis=1)
    # for i, col in enumerate(df):
    #     # using i allows for duplicate columns
    #     # inplace *may* not always work here, so IMO the next line is preferred
    #     # df.iloc[:, i].fillna(m, inplace=True)
    #     df.iloc[:, i] = df.iloc[:, i].fillna(m)
    # df.to_csv(f"{other_directory}parition2_r_values_totusjz_totusjh_mean.csv")
    #
    # exit(1)

    names = [
        "KNN",
        "RFC",
        "LR",
        "SVM",
    ]

    classifiers = [
        KNeighborsClassifier(n_neighbors=3),
        RandomForestClassifier(n_estimators=120),
        LogisticRegression(C=1000),
        SVC(),
    ]
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    train = pd.read_csv(f"{other_directory}parition1_r_values_totusjz_totusjh_mean.csv")
    test = pd.read_csv(f"{other_directory}parition2_r_values_totusjz_totusjh_mean.csv")
    # train = train.loc[~train["FLARE_TYPE"].str.contains("C")]
    # test = test.loc[~test["FLARE_TYPE"].str.contains("C")]
    # for flare_type in ["B", "C", "M", "X"]:
    #     print(f"{flare_type}: {train.loc[train['FLARE_TYPE'].str.contains(flare_type)].shape[0]}")
    # for flare_type in ["B", "C", "M", "X"]:
    #     print(f"{flare_type}: {test.loc[test['FLARE_TYPE'].str.contains(flare_type)].shape[0]}")

    train = train.drop("Unnamed: 0", axis=1)
    train["LABEL"] = train["FLARE_TYPE"].apply(get_ar_class)
    test = test.drop("Unnamed: 0", axis=1)
    test["LABEL"] = test["FLARE_TYPE"].apply(get_ar_class)

    for feats in powerset(["R_VALUE", "TOTUSJH", "TOTUSJZ"]):
        features = [f"{param}_{i}" for param in feats for i in range(1, 61) ]
        train_X, train_y = train[features], train["LABEL"].values
        test_X, test_y = test[features], test["LABEL"].values
        print(feats)
        for clf, name in zip(classifiers, names):
            clf.fit(train_X, train_y)
            y_pred = clf.predict(test_X)
            tss = calc_tss(test_y, y_pred)
            print(f"{name}: {tss}")
        print()


    # Partition1 Train Counts
    # X: 165
    # M: 1089
    # B: 5692
    # C: 3054

    # Partition2 Test Counts
    # B: 4978
    # C: 3621
    # M: 1329
    # X: 72






if __name__ == "__main__":
    main()
