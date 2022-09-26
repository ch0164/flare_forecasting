################################################################################
# Filename: experiment_template.py
# Description: This file is a correlation for easy creation of new experiments.
################################################################################

# Custom Imports
from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "generate_time_series_means"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for today for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    # ------------------------------------------------------------------------
    times = [(0, 24), (6, 18), (6, 22), (6, 24), (10, 22), (12, 24)]
    flare_classes = ["B", "MX"]
    orig_dfs = []
    modi_dfs = []
    for flare_class in flare_classes:
        for lo_time, hi_time in times:
            orig_dfs.append(get_ar_properties(flare_class,
                                              lo_time,
                                              hi_time,
                                              coincidence_time_window="0h_24h"))
            modi_dfs.append(get_ar_properties(flare_class,
                                              lo_time,
                                              hi_time))

    # ------------------------------------------------------------------------
    # Generate the figures of this experiment.
    # Afterwards, place these figures in `figure_directory`.
    # --- [ENTER CODE HERE]

    # ------------------------------------------------------------------------
    # Generate other kinds of output for the experiment.
    # Afterwards, place the output in `other_directory`.
    # --- [ENTER CODE HERE]


if __name__ == "__main__":
    main()
