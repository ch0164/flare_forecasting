# Custom Imports
import pandas as pd

from source.utilities import *

# Experiment Name (No Acronyms)
experiment = "classification_comparison"
experiment_caption = experiment.title().replace("_", " ")
# ------------------------------------------------------------------------
# Place any results in the directory for the current experiment.
now_string, figure_directory, metrics_directory, other_directory = \
    build_experiment_directories(experiment)

def classification_plot():
    df = pd.DataFrame(columns=["name", "score", "performance"])
    names = ["KNN", "RFC", "LR", "SVM"]
    df = pd.read_csv(f"{metrics_directory}bc_mx_classification_comparison.csv")



    ax = sns.barplot(data=df, x="name", y="tss", hue="data")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_title("BC vs. MX Classification Comparison, All Flares")
    ax.set_ylim(bottom=0.0, top=1.0)
    plt.tight_layout()
    plt.show()
    plt.clf()



def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    classification_plot()



if __name__ == "__main__":
    main()
