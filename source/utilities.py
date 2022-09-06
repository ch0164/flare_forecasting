################################################################################
# Filename: utilities.py
# Description: This file contains utility functions common to each experiment.
################################################################################

from source.common_imports import *


# ---------------------------------------------------------------------------
# --- Flare Functions

def classify_flare(magnitude: str) -> str:
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
        return "M"
    else:
        return "X"


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
    elif "data" in filename:
        df = pd.read_csv(filename, header=0, delimiter=r"\s+")
        df['T_REC'] = pd.to_datetime(df['T_REC'],
                                     format='%Y.%m.%d_%H:%M:%S.000_TAI')
    else:
        raise Exception(f"Error: {filename} is not a flare list/data file.")

    return df


def filter_data(df: pd.DataFrame(),
                nar: int,
                time_range_lo: datetime,
                time_range_hi: datetime) -> pd.DataFrame():
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

    is_good_data = (df["T_REC"] <= time_range_hi) & \
                   (df["T_REC"] >= time_range_lo) & \
                   (df["NOAA_AR"] == nar) & \
                   (df["QUALITY"] == 0) & \
                   (abs(0.5 * (df["LONMIN"] + df["LONMAX"])) <= 60) & \
                   (abs(0.5 * (df["LATMIN"] + df["LATMAX"])) <= 60)

    temp_df = df.where(is_good_data).dropna()
    return temp_df.replace(0, np.NaN)


def get_ar_properties(flare_class: str,
                      needed_where: int,
                      average_over: int) -> pd.DataFrame():
    """
    Args:
        flare_class: The flare class used to partition the dataset.
        needed_where: How much ahead AR properties are needed (in hours).
        average_over: What should be the averaging window centered on this time
                      (in hours).

    Returns:
        A filtered dataframe with corresponding to the flare list of the same
        class, with AR properties appended as columns to each "good" flare.
    """
    flare_list_df = get_dataframe(
        f"{FLARE_LIST_DIRECTORY}{flare_class.lower()}_list.txt")
    flare_data_df = get_dataframe(
        f"{FLARE_DATA_DIRECTORY}{flare_class.lower()}_data.txt")

    for index, row in flare_list_df.iterrows():
        print(f"Computing {flare_class} Flare {index}/{flare_list_df.shape[0]}")
        nar = row['nar']
        time_range_lo = row['time_start'] - \
            timedelta(hours=(needed_where + average_over / 2))
        time_range_hi = row['time_start'] - \
            timedelta(hours=(needed_where - average_over / 2))

        needed_slice = filter_data(flare_data_df,
                                   nar,
                                   time_range_lo,
                                   time_range_hi)
        needed_slice_avg = pd.DataFrame([needed_slice.mean(axis=0)])

        for column in flare_data_df.columns:
            if column not in ["T_REC", "NOAA_AR", "QUALITY"] + LLA_HEADERS:
                flare_list_df.loc[index, column] = needed_slice_avg.loc[
                    0, column]

    return flare_list_df


# ---------------------------------------------------------------------------
# --- Directory functions

def build_experiment_directories(experiment) -> (str, str, str, str):
    """
    Args:
        experiment: The name of the experiment being tested.

    Returns:
        A tuple of strings being:
        1. The current date and time in string format.
        2. The "cleaned_data" directory for the experiment.
        3. The "figures" directory for the experiment.
        4. The "other" directory for the experiment.
    """
    now_time = datetime.now()
    now_string = now_time.strftime("%d_%m_%Y_%H_%M_%S")
    today_string = now_time.strftime("%d_%m_%Y")

    today_directory = f"{RESULTS_DIRECTORY}{today_string}/"
    experiment_directory = f"{today_directory}{experiment}/"
    cleaned_data_directory = f"{experiment_directory}{CLEANED_DATA}/"
    figure_directory = f"{experiment_directory}{FIGURES}/"
    other_directory = f"{experiment_directory}{OTHER}/"

    if today_string not in os.listdir(RESULTS_DIRECTORY):
        os.mkdir(today_directory)
    if experiment not in os.listdir(today_directory):
        os.mkdir(experiment_directory)
    if CLEANED_DATA not in os.listdir(experiment_directory):
        os.mkdir(cleaned_data_directory)
    if FIGURES not in os.listdir(experiment_directory):
        os.mkdir(figure_directory)
    if OTHER not in os.listdir(experiment_directory):
        os.mkdir(other_directory)

    return now_string, cleaned_data_directory, figure_directory, other_directory


# ---------------------------------------------------------------------------
# --- General Functions


