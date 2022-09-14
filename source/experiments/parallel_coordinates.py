################################################################################
# Filename: parallel_coordinates.py
# Description: Todo
################################################################################

# Custom Imports
from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "parallel_coordinates"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for today for the current experiment.
    now_string, cleaned_data_directory, figure_directory, other_directory = \
        build_experiment_directories(experiment)

    # ------------------------------------------------------------------------
    # Pre-process the data for the experiment.
    # Afterwards, place this data in `cleaned_data_directory`.
    time_interval = 12
    lo_time = 10
    hi_time = lo_time + time_interval
    flare_dataframes = [
        get_ar_properties(flare_class,
                          lo_time,
                          hi_time,
                          cleaned_data_directory,
                          now_string,
                          wipe_old_data=False
                          )
        for flare_class in FLARE_CLASSES
    ]

    for coincidence in COINCIDENCES:
        fig, ax = plt.subplots(figsize=(28, 22))
        for flare_class, flare_df, color in zip(FLARE_CLASSES, flare_dataframes, FLARE_COLORS):
            df = flare_df
            if flare_class in "NULL":
                if coincidence == "coincident":
                    continue
                else:
                    flare_df["xray_class"] = "N"
            elif coincidence == "coincident":
                df = flare_df.loc[flare_df["COINCIDENCE"] == True]
            elif coincidence == "noncoincident":
                df = flare_df.loc[flare_df["COINCIDENCE"] == False]

            properties_df = df[FLARE_PROPERTIES].dropna()
            # normalized_df = (properties_df - properties_df.mean()) / properties_df.std()
            normalized_df = (properties_df - properties_df.min()) / (properties_df.max() - properties_df.min())

            normalized_df["xray_class"] = df["xray_class"]
            parallel_coordinates(normalized_df, "xray_class", FLARE_PROPERTIES, ax, color=color, axvlines=True)

        # fig.legend()
        # fig.title(f"{coincidence.capitalize()} {experiment_caption} for NULL/B/MX Flares")
        fig.show()



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
