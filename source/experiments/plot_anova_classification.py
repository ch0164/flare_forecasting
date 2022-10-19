from matplotlib import pyplot as plt
import json
from source.common_imports import *

names = [
        "LDA",
        "KNN 3",
        "Logistic Regression",
        "Linear SVM",
        "Random Forest",
    ]

def main():
    for time_window in ["12h_13h"]:
        file = f"{RESULTS_DIRECTORY}classification/other/{time_window}_anova.json"
        fp = open(file)
        clf_scores = json.load(fp)
        fp.close()

        x = range(1, len(clf_scores["LDA"]) + 1)
        for name in names:
            y = clf_scores[name]
            plt.plot(x, y, label=name)
        plt.xticks(x)
        plt.legend()
        plt.title(f"ANOVA Parameters for NB/MX Flare Classification, {time_window}")
        plt.xlabel("# of Top ANOVA Parameters")
        plt.ylabel("True Skill Score (TSS)")
        plt.show()

if __name__ == "__main__":
    main()