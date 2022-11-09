################################################################################
# Filename: pearson_correlation.py
# Description: This file is a correlation for easy creation of new experiments.
################################################################################

# Custom Imports
import pandas as pd

from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "pearson_correlation"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for today for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    flare_classes = ["NB", "MX"]

    for lo_time, hi_time in [(0, 24), (5, 17)]:

        # Get the time window of the experiment for metadata.
        time_window = get_time_window(lo_time, hi_time)
        time_window_caption = time_window.replace("_", "-")

        # Obtain the properties for flares.

        flare_dataframes = [
            get_ar_properties(flare_class,
                              lo_time,
                              hi_time,
                              coincidence_flare_classes="nbmx",
                              # timepoint=13*24 + 24,
                              # coincidence_time_window="0h_24h"
                              )
            for flare_class in flare_classes
        ]
        for coincidence in ["all", "coincident", "noncoincident"]:
            if coincidence == "coincident":
                is_coincident = True
            elif coincidence == "noncoincident":
                is_coincident = False
            else:
                is_coincident = None
            all_flares_df = pd.concat(flare_dataframes).dropna()
            if is_coincident is not None:
                all_flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == is_coincident]
            all_flares_df = all_flares_df. \
                reset_index(). \
                drop(["index"], axis=1). \
                rename_axis("index")

            # params = [
            #     "TOTUSJH",
            #     "USFLUX",
            #     "TOTUSJZ",
            #     "R_VALUE",
            #     "TOTPOT",
            #     "AREA_ACR",
            #     "SAVNCPP",
            #     "ABSNJZH",
            #     "MEANPOT",
            #     "SHRGT45",
            # ]
            cm = all_flares_df[FLARE_PROPERTIES].corr()
            sns.heatmap(cm,
                        xticklabels=cm.columns.values,
                        yticklabels=cm.columns.values,
                        cmap="coolwarm")
            plt.title(f"{'/'.join(flare_classes)} {coincidence} Flares Correlation Matrix, {time_window_caption},\n"
                      f"All Flare Parameters")
            plt.savefig(f"{figure_directory}{coincidence}/{'_'.join(flare_classes).lower()}_correlation_matrix_on_sinha_parameters_{time_window}.png")
            plt.show()


        # flare_df["xray_class"].replace("N", "NB", inplace=True)
        # flare_df["xray_class"].replace("B", "NB", inplace=True)
        # X = flare_df[FLARE_PROPERTIES]
        # y = flare_df["xray_class"]


if __name__ == "__main__":
    main()
