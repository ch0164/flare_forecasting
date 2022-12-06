################################################################################
# Filename: experiment_template.py
# Description: This file is a correlation for easy creation of new experiments.
################################################################################

# Custom Imports
import pandas as pd

from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "correlation"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for today for the current experiment.
    now_string, cleaned_data_directory, figure_directory, other_directory = \
        build_experiment_directories(experiment)

    # ------------------------------------------------------------------------
    # Pre-process the data for the experiment.
    # Afterwards, place this data in `cleaned_data_directory`.
    lo_time = 21
    hi_time = 23

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
        for flare_class in ["NB", "MX"]
    ]
    flare_df = pd.concat(flare_dataframes).dropna()
    flare_df["xray_class"].replace("N", "NB", inplace=True)
    flare_df["xray_class"].replace("B", "NB", inplace=True)
    X = flare_df[FLARE_PROPERTIES]
    y = flare_df["xray_class"]

    f = pd.DataFrame(f_classif(X, y), columns=FLARE_PROPERTIES).iloc[0]
    f = f.values.reshape(-1, 1)

    f_n = MinMaxScaler().fit_transform(f).ravel()
    f_n = pd.Series(f_n, index=FLARE_PROPERTIES).sort_values(ascending=False)
    f = pd.Series(f.ravel(), index=FLARE_PROPERTIES).sort_values(ascending=False)
    f_df = pd.DataFrame({"f_score": f, "f_score_norm": f_n}).rename_axis("parameter")
    f_df.to_csv(f"{other_directory}nb_mx_anova_f_scores_{time_window}.csv")

    # X[X < 0] = 0
    # chi = pd.DataFrame(chi2(X, y), columns=FLARE_PROPERTIES).iloc[0]
    # chi = chi.values.reshape(-1, 1)
    #
    # chi_n = MinMaxScaler().fit_transform(chi).ravel()
    # chi_n = pd.Series(chi_n, index=FLARE_PROPERTIES).sort_values(ascending=False)
    # chi = pd.Series(chi.ravel(), index=FLARE_PROPERTIES).sort_values(
    #     ascending=False)
    # chi_df = pd.DataFrame({"chi2_score": chi, "chi2_score_norm": chi_n}).\
    #     rename_axis("parameter")
    # chi_df.to_csv(
    #     f"{other_directory}nbmx_chi2_scores_{time_window}.csv")


if __name__ == "__main__":
    main()
