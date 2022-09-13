################################################################################
# Filename: coincidence.py
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

    # ------------------------------------------------------------------------
    # Determine the time window on which to calculate flare coincidence.
    time_interval = 24
    lo_time = 0
    hi_time = lo_time + time_interval
    lo_time_complement = 24 - lo_time
    hi_time_complement = 24 - hi_time

    # Get the time window of the experiment for metadata.
    time_window = f"{lo_time}h_{hi_time}h"
    time_window_caption = time_window.replace("_", "-")

    # Combine all classes of flares (except NULL flares) into one dataframe.
    flare_list_df = pd.DataFrame()
    for flare_class in FLARE_CLASSES:
        if flare_class != "NULL":
            flare_list_df = pd.concat(
                [flare_list_df,
                get_dataframe(f"{FLARE_LIST_DIRECTORY}{flare_class}_list.txt")
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
        print(f"Flare {index}/{flare_list_df.shape[0]}")
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
    if time_window not in os.listdir(FLARE_LIST_DIRECTORY):
        os.mkdir(FLARE_LIST_DIRECTORY + time_window)

    b_list_df = flare_list_df.loc[flare_list_df["xray_class"] == "B"]
    b_list_df.reset_index(inplace=True)
    b_list_df.drop("index", axis=1, inplace=True)
    b_list_df.rename_axis("index", inplace=True)
    b_list_df.to_csv(f"{FLARE_LIST_DIRECTORY}{time_window}/b_list.txt")

    mx_list_df = flare_list_df.loc[(flare_list_df["xray_class"] == "M") |
                                  (flare_list_df["xray_class"] == "X")]
    mx_list_df.reset_index(inplace=True)
    mx_list_df.drop("index", axis=1, inplace=True)
    mx_list_df.rename_axis("index", inplace=True)
    mx_list_df.to_csv(f"{FLARE_LIST_DIRECTORY}{time_window}/mx_list.txt")


if __name__ == "__main__":
    main()
