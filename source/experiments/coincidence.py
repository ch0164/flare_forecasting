################################################################################
# Filename: coincident.py
# Description: Todo
################################################################################

# Custom Imports
from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

    # Experiment Name (No Acronyms)
    experiment = "coincidence"

    flare_classes = ["NB", "MX"]

    # ------------------------------------------------------------------------
    # Generate all possible time windows.
    # Only looking at 1h, 2h, 4h, 8h, 12h, 24h time windows.
    time_intervals = [1, 2, 4, 8, 12, 24]
    for time_interval in time_intervals:
        possible_time_windows = [
            (start_time, end_time) for start_time in range(0, 24)
            for end_time in range(1, 25)
            if start_time < end_time and
               abs(end_time - start_time) == time_interval
        ]


        # For each time window, find the coincident.
        for index, (lo_time, hi_time) in enumerate(possible_time_windows):
            time_window = f"{lo_time}h_{hi_time}h"
            lo_time_complement = 24 - lo_time
            hi_time_complement = 24 - hi_time

            print(f"Time window {time_window},"
                  f" time interval {time_interval}, "
                  f"{index}/{len(possible_time_windows)}")
            # if time_window in os.listdir(COINCIDENCE_LIST_DIRECTORY):
            #     continue

            # Combine all classes of flares (except NULL flares) into one dataframe.
            flare_list_df = pd.DataFrame()
            for flare_class in flare_classes:
                flare_list_df = pd.concat(
                    [flare_list_df,
                    get_dataframe(f"{FLARE_LIST_DIRECTORY}{flare_class.lower()}_list.txt")
                    ]
                )
            flare_list_df["COINCIDENCE"] = False
            flare_list_df.reset_index(inplace=True)
            flare_list_df.drop("index", axis=1, inplace=True)
            flare_list_df.rename_axis("index", inplace=True)

            # Find coincidences within the specified time window.
            # Note: Only checking that the start of one flare occurs between the start
            # and end times of a second flare's time window, where the second flare
            # is the coincident flare, regardless of flare class.
            for index, row in flare_list_df.iterrows():
                # Get this flare's info.
                # print(f"Flare {index}/{flare_list_df.shape[0]}")
                time_end = row["time_start"] - timedelta(hours=hi_time_complement)
                time_start = time_end - timedelta(hours=lo_time_complement)
                nar = row["nar"]

                # Get all the other flares in this flare's AR.
                # Then, determine if any of those flares coincide.
                flares_in_ar = flare_list_df.loc[flare_list_df["nar"] == nar]
                for index2, row2 in flares_in_ar.iterrows():
                    # Ignore the case when the flares are the same.
                    if index == index2:
                        break
                    time_start2 = row2["time_start"]
                    flares_coincide = time_start <= time_start2 <= time_end
                    if flares_coincide:
                        flare_list_df.loc[index, "COINCIDENCE"] = [True]
                        break

            # Generate the lists for the specified time window for each flare class,
            # now containing the COINCIDENCE column.
            if time_window not in os.listdir(COINCIDENCE_LIST_DIRECTORY):
                os.mkdir(COINCIDENCE_LIST_DIRECTORY + time_window)

            for flare_class in flare_classes:
                # if flare_class == "MX":
                #     break
                flare_class_df = pd.DataFrame()
                for individual_flare_class in list(flare_class):
                    # Assume null flares are noncoincident anyway.
                    # if individual_flare_class in "N":
                    #     continue
                    flare_class_df = pd.concat([
                        flare_class_df,
                        flare_list_df.loc[
                            flare_list_df["xray_class"]
                                .str.contains(individual_flare_class)]
                    ])
                flare_class_df.reset_index(inplace=True)
                flare_class_df.drop("index", axis=1, inplace=True)
                flare_class_df.rename_axis("index", inplace=True)
                flare_class_df.to_csv(
                    f"{COINCIDENCE_LIST_DIRECTORY}{time_window}/"
                    f"{flare_class.lower()}_list_nbmx.txt")


if __name__ == "__main__":
    main()
