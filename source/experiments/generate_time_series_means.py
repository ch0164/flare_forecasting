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
    time_intervals = [1, 2, 4, 8, 12, 24]
    flare_classes = ["NB", "MX"]
    for time_interval in time_intervals:
        possible_time_windows = [
            (start_time, end_time) for start_time in range(0, 24)
            for end_time in range(1, 25)
            if start_time < end_time and
               abs(end_time - start_time) == time_interval
        ]
        for flare_class in flare_classes:
            for index, (lo_time, hi_time) in enumerate(possible_time_windows):
                # mins = [minutes for minutes in range(0, 24 * 5 * 12 + 1, 12)]
                # for timepoint in mins:
                # for lo_time, hi_time in possible_time_windows:

                time_window = get_time_window(lo_time, hi_time)
                print(f"Time window {time_window},"
                      f" time interval {time_interval}, "
                      f"{index}/{len(possible_time_windows)}")
                # print(f"Timepoint={timepoint}, Flare Class={flare_class}")
                # get_ar_properties(flare_class, timepoint=timepoint,
                #                   coincidence_flare_classes="nbmx")
                get_ar_properties(flare_class,
                                  lo_time,
                                  hi_time,
                                  coincidence_flare_classes="nbmx")

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
