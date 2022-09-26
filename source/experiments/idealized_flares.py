################################################################################
# Filename: idealized_flares.py
# Description: Todo
################################################################################

# Custom Imports
from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "idealized_flares"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for today for the current experiment.
    now_string, cleaned_data_directory, figure_directory, other_directory = \
        build_experiment_directories(experiment)

    # ------------------------------------------------------------------------
    # Pre-process the data for the experiment.
    # Afterwards, place this data in `cleaned_data_directory`.
    center_time = 16
    time_interval = 12

    time_interval = 24
    lo_time = 0
    hi_time = lo_time + time_interval
    lo_time_complement = 24 - lo_time
    hi_time_complement = 24 - hi_time

    # Get the time window of the experiment for metadata.
    time_window = get_time_window(lo_time, hi_time)
    time_window_caption = time_window.replace("_", "-")

    # Obtain the properties for flares.
    # flare_dataframes = [
    #     get_ar_properties(flare_class,
    #                       lo_time,
    #                       hi_time,
    #                       cleaned_data_directory,
    #                       now_string,
    #                       wipe_old_data=False,
    #                       use_time_window=True,
    #                       coincidence_time_window="0h_24h"
    #                       )
    #     for flare_class in FLARE_CLASSES
    # ]
    #
    # # Group the flares into one dataframe.
    # all_flares_df = pd.concat(flare_dataframes).reset_index().drop(["level_0", "index"], axis=1).rename_axis("index")
    # # Null flares are assumed to already be non-coincident. Todo: Check this is really the case.
    # all_flares_df["COINCIDENCE"] = all_flares_df["COINCIDENCE"].fillna(False)
    #
    # # Split the observations by coincident, and then by flare class.
    # for coincident in COINCIDENCES:
    #     if coincident == "coincident":
    #         coincidence_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == True]
    #     elif coincident == "noncoincident":
    #         coincidence_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == False]
    #     else:
    #         coincidence_df = all_flares_df
    #
    #     for flare_class in FLARE_CLASSES:
    #         flare_df = coincidence_df.loc[coincidence_df["xray_class"] == flare_class]


    # ------------------------------------------------------------------------
    # Generate the figures of this experiment.
    # Afterwards, place these figures in `figure_directory`.
    for coincidence in COINCIDENCES:
        fig, ax = plt.subplots(5, 3, figsize=(20, 22))
        for flare_class, color in zip(FLARE_CLASSES, FLARE_COLORS):
            if flare_class == "NULL" and coincidence == "coincident":
                continue
            # df_ave = get_idealized_flare(flare_class, coincident,
            #                              lo_time,
            #                              hi_time,
            #                              cleaned_data_directory,
            #                              now_string,
            #                              wipe_old_data=False,
            #                              use_time_window=True,
            #                              coincidence_time_window="0h_24h"
            #                              )

            df_ave = pd.read_csv(f"{cleaned_data_directory}{coincidence.lower()}_{flare_class.lower()}_{time_window}_idealized_flare_19_09_2022_10_04_58.csv")

            # Plot specified flare properties over the specified time.
            row, col = 0, 0
            # df_ave.dropna(inplace=True)
            to_drop = ["MEANGBH", "MEANGBT", "MEANGBZ", "MEANJZD", "slf"]
            new_flare_properties = FLARE_PROPERTIES[:]
            for prop in to_drop:
                new_flare_properties.remove(prop)

            for flare_property in new_flare_properties:
                property_df = df_ave[[flare_property]]
                property_np = property_df.to_numpy().ravel()
                std_error = np.std(property_np, ddof=0) / np.sqrt(len(property_np))
                # property_df.plot(y=flare_property, ax=ax[row, col], color=color,
                #                  label=label)
                ax[row, col].errorbar(x=range(len(property_np)), y=property_np,
                                      yerr=std_error, capsize=4, color=color, label=flare_class)
                ax[row, col].set_ylabel(flare_property)
                ax[row, col].set_title(f"{flare_property}")
                ax[row, col].legend()

                col += 1
                if col == 3:
                    col = 0
                    row += 1


        fig.tight_layout()
        plt.savefig(f'{figure_directory}/{coincidence}_null_b_mx_{time_window}_idealized_flare_{now_string}.png')
        fig.show()


    # ------------------------------------------------------------------------
    # Generate other kinds of output for the experiment.
    # Afterwards, place the output in `other_directory`.
    # --- [ENTER CODE HERE]


if __name__ == "__main__":
    main()
