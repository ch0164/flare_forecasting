################################################################################
# Filename: multiple_ars.py
# Description: Todo
################################################################################

# Custom Imports
from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "multiple_ars"
    experiment_caption = experiment.title().replace("_", " ")

    # Place any results in the directory for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)




if __name__ == "__main__":
    main()
