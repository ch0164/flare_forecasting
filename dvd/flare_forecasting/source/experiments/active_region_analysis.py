################################################################################
# Filename: active_region_analysis.py
# Description: A simple analysis to test out the newly obtained data.
################################################################################

# Custom Imports
from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "active_region_analysis"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for today for the current experiment.
    now_string, cleaned_data_directory, figure_directory, other_directory = \
        build_experiment_directories(experiment)

    # ------------------------------------------------------------------------
    # Pre-process the data for the experiment.
    # Afterwards, place this data in `cleaned_data_directory`.

    # Describe what time period of the 24h time-series data to use
    # before flare onset, then generate dataframes with the properties of flares
    # in that period.
    needed_where = 12
    average_over = 6
    flare_dataframes = [
        get_ar_properties(flare_class, needed_where, average_over)
        for flare_class in FLARE_CLASSES
    ]

    # Get the time window of the experiment for metadata.
    lo_time = int(24 - (needed_where + average_over // 2))
    hi_time = int(24 - (needed_where - average_over // 2))
    time_window = f"{lo_time}h_{hi_time}h"
    time_window_caption = time_window.replace("_", "-")

    # Save our data.
    for flare_class, df in zip(FLARE_CLASSES, flare_dataframes):
        df.to_csv(f"{cleaned_data_directory}{flare_class.lower()}_"
                  f"{time_window}_mean_dataset_{now_string}.csv")

    # ------------------------------------------------------------------------
    # Generate the figures of this experiment.
    # Afterwards, place these figures in `figure_directory`.
    rows, cols = 5, 4
    f, ax = plt.subplots(rows, cols, figsize=LANDSCAPE_FIGSIZE)
    for flare_class, flare_color, df in \
            zip(FLARE_CLASSES, FLARE_COLORS, flare_dataframes):
        # Plot the means of all flare properties.
        i, j = 0, 0
        for flare_property in FLARE_PROPERTIES:
            sns.scatterplot(data=df,
                            x="time_start",
                            y=flare_property,
                            ax=ax[i, j],
                            color=flare_color,
                            alpha=0.7)
            j += 1
            if j == cols:
                j = 0
                i += 1

    # Add title and save the plot.
    plt.suptitle(f"{experiment_caption} for {', '.join(FLARE_CLASSES)} Flares "
                 f"from {time_window_caption} Mean",
                 fontsize=LANDSCAPE_TITLE_FONTSIZE,
                 y=0.99)
    plt.tight_layout()
    plt.savefig(
        f"{figure_directory}scatterplot_{time_window}_mean_{now_string}.png")
    plt.show()

    # ------------------------------------------------------------------------
    # Generate other kinds of output for the experiment.
    # Afterwards, place the output in `other_directory`.


if __name__ == "__main__":
    main()
