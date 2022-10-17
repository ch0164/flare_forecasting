################################################################################
# Filename: utilities.py
# Description: This file contains utility functions common to each experiment.
################################################################################
import os.path

from source.common_imports import *


# ---------------------------------------------------------------------------
# --- Flare Functions

def classify_flare(magnitude: str, combine: bool = False) -> str:
    """
    Args:
        magnitude: The magnitude of a flare (e.g., X6.1).

    Returns:
        Converts magnitude of a flare to a single letter
        for use in classification.
    """
    if "B" in magnitude:
        return "B"
    elif "C" in magnitude:
        return "C"
    elif "M" in magnitude:
        if combine:
            return "MX"
        return "M"
    elif "X" in magnitude:
        if combine:
            return "MX"
        return "X"
    else:
        return "N"


def get_time_window(lo_time: int = 10, hi_time: int = 22):
    """
    Args:
        lo_time:
        hi_time:

    Returns:

    """
    return f"{lo_time}h_{hi_time}h"


def get_time_series_means_filename(flare_class: str, time_window: str,
                                   coincidence_definition: str,
                                   coincidence_flare_classes: str = "") -> str:
    """
    Args:
        flare_class:
        time_window:
        coincidence_definition:

    Returns: If there is data at the cleaned_data_directory for the experiment,
             this function returns the first dataset that

    """
    dir = f"{FLARE_MEANS_DIRECTORY}{coincidence_definition}/{time_window}/"
    if coincidence_flare_classes:
        substr = f"{flare_class.lower()}_{time_window}" \
                 f"_{coincidence_flare_classes}_mean_dataset"
    else:
        substr = f"{flare_class.lower()}_{time_window}_mean_dataset"
    if os.path.exists(dir):
        for file in os.listdir(dir):
            if substr in file:
                return f"{dir}{file}"
    else:
        return ""


def get_timepoint_filename(flare_class: str, timepoint: int,
                           coincidence_definition: str,
                           coincidence_flare_classes: str = "") -> str:
    """
    """
    dir = f"{FLARE_MEANS_DIRECTORY}{coincidence_definition}/{timepoint}m/"
    if coincidence_flare_classes:
        substr = f"{flare_class.lower()}_{timepoint}m" \
                 f"_{coincidence_flare_classes}_timepoint_dataset"
    else:
        substr = f"{flare_class.lower()}_{timepoint}m_timepoint_dataset"
    if os.path.exists(dir):
        for file in os.listdir(dir):
            if substr in file:
                return f"{dir}{file}"
    else:
        return ""


def get_dataframe(filename: str) -> pd.DataFrame():
    """
    Args:
        filename: The name of the flare list/data file.

    Returns:
        Reads the given list/data file and returns it as a pandas dataframe.

    Raises:
        Exception: Raised when an incorrect flare list/data file is given.
    """
    if "list" in filename:
        df = pd.read_csv(filename, header=0, index_col="index")
        df['time_start'] = pd.to_datetime(df['time_start'],
                                          format='%Y-%m-%d %H:%M:%S')
        df["xray_class"] = df["xray_class"].apply(classify_flare)
    elif "data" in filename:
        df = pd.read_csv(filename, header=0, delimiter=r"\s+")
        df['T_REC'] = pd.to_datetime(df['T_REC'],
                                     format='%Y.%m.%d_%H:%M:%S.000_TAI')
    else:
        raise Exception(f"Error: {filename} is not a flare list/data file.")

    return df


def filter_data(df: pd.DataFrame(),
                nar: int,
                time_range_lo: pd.Timestamp = None,
                time_range_hi: pd.Timestamp = None,
                timepoint: pd.Timestamp = None,
                filter_multiple_ars: bool = True) -> pd.DataFrame():
    """
    Args:
        df: The dataframe to apply a custom filter.
        nar: An active region number.
        time_range_lo: The minimum point in time allowed in the time series.
        time_range_hi: The maximum point in time allowed in the time series.

    Returns:
        A filtered/cleaned dataframe from the original dataframe passed in,
        replacing any zero values with NaN.
    """

    is_good_data = (df["NOAA_AR"] == nar) & \
                   (df["QUALITY"] == 0) & \
                   (abs(0.5 * (df["LONMIN"] + df["LONMAX"])) <= 60) & \
                   (abs(0.5 * (df["LATMIN"] + df["LATMAX"])) <= 60)

    if timepoint is not None:
        is_good_data &= (df["T_REC"] == timepoint)
    else:
        is_good_data &= (df["T_REC"] <= time_range_hi) & \
                        (df["T_REC"] >= time_range_lo)

    # If the flare has multiple ARs, ignore them.
    if filter_multiple_ars:
        is_good_data &= ~df["ARs"].str.contains(",")

    return df.where(is_good_data).dropna()


def get_idealized_flare(flare_class: str,
                        coincidence: str,
                        lo_time: int = 10,
                        hi_time: int = 22,
                        cleaned_data_directory: str = "",
                        now_string: str = "",
                        wipe_old_data: bool = False,
                        use_time_window: bool = True,
                        coincidence_time_window: str = "") -> pd.DataFrame():
    """Todo: The description
    Args:
        flare_class: The flare class used to partition the dataset.
        lo_time: The beginning of the time window for the time series
                 before flare onset.
                 Valid values are in range 0-24.
        hi_time: The end of the time window for the time series
                 before flare onset.
                 Valid values are in range 0-24.
        cleaned_data_directory: The directory which contains

    Returns:
        A filtered dataframe with corresponding to the flare list of the same
        class, with AR properties appended as columns to each "good" flare.
    """
    #
    time_window = get_time_window(lo_time, hi_time)
    time_interval = hi_time - lo_time

    # if wipe_old_data:
    #     for file in os.listdir(cleaned_data_directory):
    #         if now_string not in file:
    #             if use_time_window and time_window not in file:
    #                 os.remove(f"{cleaned_data_directory}{file}")
    #
    # # Determine if data already exists for this flare class and time window.
    # flare_dataset_file = get_cleaned_data_filename(flare_class,
    #                                                time_window,
    #                                                cleaned_data_directory)
    # # If so, then don't compute anything and return this data instead.
    # if flare_dataset_file:
    #     flare_list_df = pd.read_csv(flare_dataset_file)
    #     return flare_list_df

    if flare_class != "N" and use_time_window:
        if coincidence_time_window:
            filename = f"{FLARE_LIST_DIRECTORY}{coincidence_time_window}/" \
                       f"{flare_class.lower()}_list.txt"
        else:
            filename = f"{FLARE_LIST_DIRECTORY}{time_window}/" \
                       f"{flare_class.lower()}_list.txt"
    else:
        filename = f"{FLARE_LIST_DIRECTORY}{flare_class.lower()}_list.txt"

    flare_list_df = get_dataframe(filename)
    if coincidence == "coincident":
        flare_list_df = flare_list_df.loc[flare_list_df["COINCIDENCE"] == True]
    elif coincidence == "noncoincident":
        flare_list_df = flare_list_df.loc[flare_list_df["COINCIDENCE"] == False]
    flare_list_df["xray_class"] = flare_list_df["xray_class"].apply(classify_flare)
    flare_data_df = get_dataframe(
        f"{FLARE_DATA_DIRECTORY}{flare_class.lower()}_data.txt")

    df_1_sum = pd.DataFrame(columns=FLARE_PROPERTIES)
    df_2_sum = pd.DataFrame(columns=FLARE_PROPERTIES)
    for flare_property in FLARE_PROPERTIES:
        df_1_sum[flare_property] = np.zeros(time_interval * 5)
        df_2_sum[flare_property] = np.zeros(time_interval * 5)

    progress_index = 0
    for index, row in flare_list_df.iterrows():
        progress_index += 1
        print(f"Computing {coincidence} {flare_class} Flare {progress_index}/{flare_list_df.shape[0]}")
        nar = row['nar']
        time_range_lo = row['time_start'] - timedelta(hours=hi_time)
        time_range_hi = row['time_start'] - timedelta(hours=lo_time)

        needed_slice = filter_data(flare_data_df,
                                   nar,
                                   time_range_lo,
                                   time_range_hi)

        if needed_slice.empty:
            continue
        start_index, end_index = needed_slice.index[0], needed_slice.index[-1]
        df_1 = pd.DataFrame(columns=FLARE_PROPERTIES)
        df_2 = pd.DataFrame(columns=FLARE_PROPERTIES)
        for flare_property in FLARE_PROPERTIES:
            df_1[flare_property] = np.zeros(time_interval * 5)
            df_2[flare_property] = np.zeros(time_interval * 5)
            for i in range(time_interval * 5 - 1, -1, -1):
                local_df_ind = end_index - (time_interval * 5 - 1 - i)
                if local_df_ind not in needed_slice.index:
                    continue
                if local_df_ind >= 0 and local_df_ind >= start_index:
                    df_1.at[i, flare_property] = needed_slice.at[
                        local_df_ind, flare_property]
                if df_1.at[i, flare_property] != 0:
                    df_2.at[i, flare_property] = 1

        # needed_slice.loc[:, 'xray_class'] = flare_class
        # needed_slice.loc[:, 'time_start'] = time_range_lo
        # local_properties_df.loc[:, 'flare_index'] = flare_index
        # df_needed = pd.concat([df_needed, local_properties_df])

        df_1_sum = df_1_sum.add(df_1)
        df_2_sum = df_2_sum.add(df_2)

    df_ave = df_1_sum.div(df_2_sum)

    # needed_slice_avg = pd.DataFrame([needed_slice.mean(axis=0)])
    #
    # for column in flare_data_df.columns:
    #     if column not in ["T_REC", "NOAA_AR", "QUALITY"] + LLA_HEADERS:
    #         flare_list_df.loc[index, column] = needed_slice_avg.loc[
    #             0, column]

    # Save the dataset before exiting this function.
    filename = f"{cleaned_data_directory}" + \
               f"{coincidence}_{flare_class.lower()}_{time_window}_idealized_flare_" + \
               f"{now_string}.csv"
    df_ave.to_csv(filename)

    return df_ave


def get_ar_properties(flare_class: str,
                      lo_time: Any = 0,
                      hi_time: Any = 0,
                      timepoint: Any = None,
                      coincidence_time_window: str = "0h_24h",
                      coincidence_flare_classes: str = "",
                      filter_multiple_ars: bool = True) -> pd.DataFrame():
    """
    Args:
        flare_class: The flare class used to partition the dataset.
        lo_time: The beginning of the time window for the time series
                 before flare onset.
                 Valid values are in range 0-24.
        hi_time: The end of the time window for the time series
                 before flare onset.
                 Valid values are in range 0-24.
        cleaned_data_directory: The directory which contains

    Returns:
        A filtered dataframe with corresponding to the flare list of the same
        class, with AR properties appended as columns to each "good" flare.
    """
    # Determine if data already exists for this flare class and time window.
    time_window = get_time_window(lo_time, hi_time)

    if coincidence_time_window == "0h_24h":
        coincidence_definition = "original_coincidence_definition"
    else:
        coincidence_definition = "modified_coincidence_definition"

    if timepoint is not None:
        flare_dataset_file = get_timepoint_filename(flare_class,
                                                    timepoint,
                                                    coincidence_definition,
                                                    coincidence_flare_classes)
    else:
        flare_dataset_file = get_time_series_means_filename(
            flare_class,
            time_window,
            coincidence_definition,
            coincidence_flare_classes
        )

    # If so, then don't compute anything and return this data instead.
    if flare_dataset_file:
        if coincidence_flare_classes:
            flare_dataset_file = flare_dataset_file.replace("_list.txt",
                                       f"_list_{coincidence_flare_classes}.txt")
        flare_list_df = pd.read_csv(flare_dataset_file)
        return flare_list_df

    # Get the flare info list file.
    if flare_class != "N":
        if coincidence_time_window:
            filename = f"{COINCIDENCE_LIST_DIRECTORY}{coincidence_time_window}/" \
                       f"{flare_class.lower()}_list.txt"
        else:
            filename = f"{COINCIDENCE_LIST_DIRECTORY}{time_window}/" \
                       f"{flare_class.lower()}_list.txt"
    else:
        filename = f"{FLARE_LIST_DIRECTORY}{flare_class.lower()}_list.txt"

    if coincidence_flare_classes:
        filename = filename.replace("_list.txt",
                                   f"_list_{coincidence_flare_classes}.txt")

    # Read the flare list into a dataframe and clean it.
    flare_list_df = get_dataframe(filename)
    flare_list_df["xray_class"] = flare_list_df["xray_class"].apply(classify_flare)

    # Get the corresponding data for this flare class.
    flare_data_df = get_dataframe(
        f"{FLARE_DATA_DIRECTORY}{flare_class.lower()}_data.txt")

    # Get the AR params for a single timepoint.
    if timepoint is not None:
        for index, row in flare_list_df.iterrows():
            print(f"{index}/{flare_list_df.shape[0]}")
            def floor_minute(time, cadence=12):
                return time - timedelta(minutes=time.minute % cadence)
            nar = row["nar"]
            timestamp = row["time_start"] - timedelta(minutes=timepoint)
            timestamp = floor_minute(timestamp)
            needed_slice = filter_data(flare_data_df,
                                       nar,
                                       timepoint=timestamp).reset_index()
            if needed_slice.empty:
                continue
            for column in flare_data_df.columns:
                if column not in ["T_REC", "NOAA_AR", "QUALITY", "ARs"] + LLA_HEADERS:
                    # print(flare_list_df.to_string())
                    flare_list_df.loc[index, column] = needed_slice.loc[
                        0, column]

        dir = f"{FLARE_MEANS_DIRECTORY}{coincidence_definition}/"
        timepoint_caption = f"{timepoint}m"
        if timepoint_caption not in os.listdir(dir):
            os.mkdir(dir + timepoint_caption)
        filename = f"{dir + timepoint_caption}/" \
                   f"{flare_class.lower()}_{timepoint_caption}" \
                   f"{'_' + coincidence_flare_classes}_timepoint_dataset.csv"
        flare_list_df.to_csv(filename)

        return flare_list_df

    # Get the AR params for a time series.
    for index, row in flare_list_df.iterrows():
        # print(f"Computing {flare_class} Flare {index}/{flare_list_df.shape[0]}")
        nar = row['nar']

        time_range_lo = row['time_start'] - timedelta(hours=hi_time)
        time_range_hi = row['time_start'] - timedelta(hours=lo_time)

        needed_slice = filter_data(flare_data_df,
                                   nar,
                                   time_range_lo,
                                   time_range_hi,
                                   filter_multiple_ars=filter_multiple_ars)
        needed_slice_avg = pd.DataFrame([needed_slice[FLARE_PROPERTIES].mean(axis=0)])
        multiple_ars = needed_slice.loc[needed_slice["ARs"].str.contains(",")]
        if multiple_ars.empty:
            needed_slice_avg["ARs"] = ["singular"]
        else:
            needed_slice_avg["ARs"] = ["multiple"]

        for column in flare_data_df.columns:
            if column not in ["T_REC", "NOAA_AR", "QUALITY"] + LLA_HEADERS:
                flare_list_df.loc[index, column] = needed_slice_avg.loc[
                    0, column]

    # Save the dataset before exiting this function.
    dir = f"{FLARE_MEANS_DIRECTORY}{coincidence_definition}/"
    if time_window not in os.listdir(dir):
        os.mkdir(dir + time_window)
    filename = f"{dir + time_window}/" \
               f"{flare_class.lower()}_{time_window}" \
               f"_{coincidence_flare_classes}_mean_dataset.csv"
    flare_list_df.to_csv(filename)

    return flare_list_df


def write_classification_metrics(y_true: List[Any], y_pred: List[Any],
                                 filename: str,
                                 clf_name: str,
                                 flare_classes: List[str],
                                 print_output: bool = False):
    stdout = None

    cm = confusion_matrix(y_true, y_pred, labels=flare_classes)
    tn, fp, fn, tp = cm.ravel()
    detection_rate = tp / float(tp + fn)
    false_alarm_rate = fp / float(fp + tn)
    tss = detection_rate - false_alarm_rate
    cr = classification_report(y_true, y_pred, labels=flare_classes)

    if not print_output:
        stdout = sys.stdout
        sys.stdout = open(filename, "w")

    print(f"{clf_name} Classification Metrics")
    print("-" * 50)
    print("Confusion Matrix")
    print(" ", flare_classes[0], " ", flare_classes[1])
    print(flare_classes[0], cm[0])
    print(flare_classes[1], cm[1])
    print("-" * 50)
    print("Classification Report")
    print(cr)
    print(f"True Skill Score: {tss:.4f}")

    if not print_output:
        sys.stdout.close()
        sys.stdout = stdout
# ---------------------------------------------------------------------------




# --- Directory functions

def build_experiment_directories(experiment: str) -> (str, str, str):
    """
    Args:
        experiment: The name of the experiment being tested.

    Returns:
        A tuple of strings being:
        1. The current date and time in string format.
        2. The "figures" directory for the experiment.
        3. The "metrics" directory for the experiment.
        4. The "other" directory for the experiment.
    """
    now_time = datetime.now()
    now_string = now_time.strftime("%d_%m_%Y_%H_%M_%S")

    experiment_directory = f"{RESULTS_DIRECTORY}{experiment}/"
    figure_directory = f"{experiment_directory}{FIGURES}/"
    metrics_directory = f"{experiment_directory}{METRICS}/"
    other_directory = f"{experiment_directory}{OTHER}/"

    if experiment not in os.listdir(RESULTS_DIRECTORY):
        os.mkdir(experiment_directory)
    if FIGURES not in os.listdir(experiment_directory):
        os.mkdir(figure_directory)
    if METRICS not in os.listdir(experiment_directory):
        os.mkdir(metrics_directory)
    if OTHER not in os.listdir(experiment_directory):
        os.mkdir(other_directory)

    return now_string, figure_directory, metrics_directory, other_directory

# ---------------------------------------------------------------------------
# --- General Functions
