import pandas as pd

from source.utilities import *

def main():
    holdout_dict = {"all": [], "coincident": [], "noncoincident": []}
    loo_dict = {"all": [], "coincident": [], "noncoincident": []}

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
        "AdaBoost",
    ]

    time_window = "0h_24h"

    for cv, d in zip(["70_30_train_test", "leave_one_out"], [holdout_dict, loo_dict]):
        dir = RESULTS_DIRECTORY + "classification/metrics/" + cv + "/"
        clf_names = []
        for coincidence in COINCIDENCES:
            for file in os.listdir(dir + coincidence):
                if "bc_mx" not in file:
                    continue
                if time_window not in file:
                    continue
                if "neural_net" in file:
                    continue
                clf_name, tss = None, None
                with open(dir + coincidence + "/" + file, "r") as f:
                    for line in f:
                        if line.endswith("Classification Metrics\n"):
                            clf_name = line.split(" Classification Metrics")[0]
                            clf_names.append(clf_name)
                        delim = "True Skill Score: "
                        if line.startswith(delim):
                            tss = float(line.strip(delim))
                            d[coincidence].append(tss)
                            break
        df = pd.DataFrame(d, columns=COINCIDENCES).rename_axis("classifier")
        df.index = names
        df.to_csv(dir + f"{time_window}_bc_mx_true_skill_score_summary.csv")


if __name__ == "__main__":
    main()