import pandas as pd

from source.utilities import *

def main():
    for flare_classification in ["b_mx", "bc_mx", "b_mx_lda", "bc_mx_lda"]:
        for time_window in ["0h_24h", "10h_22h"]:
            holdout_dict = {"all": [], "coincident": [], "noncoincident": []}
            loo_dict = {"all": [], "coincident": [], "noncoincident": []}
            for cv, d in zip(["70_30_train_test", "leave_one_out"], [holdout_dict, loo_dict]):
                dir = RESULTS_DIRECTORY + "classification/metrics/" + cv + "/"
                clf_names = set()
                for coincidence in COINCIDENCES:
                    print(os.listdir(dir + coincidence))
                    for file in os.listdir(dir + coincidence):
                        if f"{flare_classification}_{time_window}" not in file:
                            continue
                        if "neural_net" in file:
                            continue
                        with open(dir + coincidence + "/" + file, "r") as f:
                            for line in f:
                                if line.endswith("Classification Metrics\n"):
                                    clf_name = line.split(" Classification Metrics")[0]
                                    clf_names.add(clf_name)
                                delim = "True Skill Score: "
                                if line.startswith(delim):
                                    tss = float(line.strip(delim))
                                    d[coincidence].append(tss)
                                    break
                df = pd.DataFrame(d, columns=COINCIDENCES).rename_axis("classifier")
                print(clf_names)
                print(df)
                df.index = clf_names
                df.rename_axis("classifier", inplace=True)
                df.to_csv(dir + f"{time_window}_{flare_classification}_true_skill_score_summary.csv")


if __name__ == "__main__":
    main()