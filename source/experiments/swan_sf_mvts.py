################################################################################
# Filename: experiment_template.py
# Description: This file is a correlation for easy creation of new experiments.
################################################################################

# Custom Imports
from source.utilities import *

# Disable Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

experiment = "swan_sf_mvts"
now_string, figure_directory, metrics_directory, other_directory = \
    build_experiment_directories(experiment)


partition1_dir = r"C:\Users\youar\Desktop\swan-sf\partition1"


class MVTSSample:
    def __init__(self, flare_type: str, start_time: datetime,
                 end_time: datetime, data: pd.DataFrame):
        self._flare_type = flare_type
        self._start_time = start_time
        self._end_time = end_time
        self._data = data

    def get_flare_type(self):
        return self._flare_type

    def get_start_time(self):
        return self._start_time

    def get_end_time(self):
        return self._end_time

    def get_data(self):
        return self._data


def read_mvts_instance(data_dir: str,
                       file_name: str) -> MVTSSample:  # Finished!
    # Get flare type from file name
    flare_type = file_name[0:2]
    start_time, end_time, data = None, None, None

    try:
        # Get start time from file name
        start = file_name.find('s2')
        start_time = file_name[start + 1: start + 20]
        start_time = start_time.replace("T", " ")
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H_%M_%S")

        # Get end time from file name
        end = file_name.find('e2')
        end_time = file_name[end + 1: end + 20]
        end_time = end_time.replace("T", " ")
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H_%M_%S")
    except ValueError:
        print(ValueError)
        pass

    # Get data from csv file
    try:
        data = pd.read_csv(data_dir + "/" + file_name, sep="\t")
    except ValueError:
        print(ValueError)
        pass

    # Make mvts object
    mvts = MVTSSample(flare_type, start_time, end_time, data)
    return mvts


def calculate_descriptive_features(data: pd.DataFrame) -> pd.DataFrame:  # Finished!
    variates_to_calc_on = ['R_VALUE', 'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ',
                           'ABSNJZH', 'SAVNCPP',
                           'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR',
                           'SHRGT45', 'MEANGAM', 'MEANGBT',
                           'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
                           'MEANALP', 'TOTFX']
    features_to_return = ['R_VALUE_MEDIAN', 'R_VALUE_STDDEV',
                          'TOTUSJH_MEDIAN', 'TOTUSJH_STDDEV',
                          'TOTBSQ_MEDIAN', 'TOTBSQ_STDDEV',
                          'TOTPOT_MEDIAN', 'TOTPOT_STDDEV',
                          'TOTUSJZ_MEDIAN', 'TOTUSJZ_STDDEV',
                          'ABSNJZH_MEDIAN', 'ABSNJZH_STDDEV',
                          'SAVNCPP_MEDIAN', 'SAVNCPP_STDDEV',
                          'USFLUX_MEDIAN', 'USFLUX_STDDEV',
                          'TOTFZ_MEDIAN', 'TOTFZ_STDDEV',
                          'MEANPOT_MEDIAN', 'MEANPOT_STDDEV',
                          'EPSZ_MEDIAN', 'EPSZ_STDDEV',
                          'MEANSHR_MEDIAN', 'MEANSHR_STDDEV',
                          'SHRGT45_MEDIAN', 'SHRGT45_STDDEV',
                          'MEANGAM_MEDIAN', 'MEANGAM_STDDEV',
                          'MEANGBT_MEDIAN', 'MEANGBT_STDDEV',
                          'MEANGBZ_MEDIAN', 'MEANGBZ_STDDEV',
                          'MEANGBH_MEDIAN', 'MEANGBH_STDDEV',
                          'MEANJZH_MEDIAN', 'MEANJZH_STDDEV',
                          'TOTFY_MEDIAN', 'TOTFY_STDDEV',
                          'MEANJZD_MEDIAN', 'MEANJZD_STDDEV',
                          'MEANALP_MEDIAN', 'MEANALP_STDDEV',
                          'TOTFX_MEDIAN', 'TOTFX_STDDEV']
    # Create empty data frame for return with named columns
    df = pd.DataFrame(columns=features_to_return)

    # For each element append to temp list
    list2add = []
    for d in variates_to_calc_on:
        l = data[d].to_numpy()
        median = np.median(l)
        std = np.std(l)
        list2add.append(median)
        list2add.append(std)
        continue

    df.loc[len(df)] = list2add
    return df


def process_partition(partition_location: str, abt_name: str):  # NEEDS WORK!
    abt_header = ['FLARE_TYPE', 'R_VALUE_MEDIAN', 'R_VALUE_STDDEV',
                  'TOTUSJH_MEDIAN', 'TOTUSJH_STDDEV',
                  'TOTBSQ_MEDIAN', 'TOTBSQ_STDDEV',
                  'TOTPOT_MEDIAN', 'TOTPOT_STDDEV',
                  'TOTUSJZ_MEDIAN', 'TOTUSJZ_STDDEV',
                  'ABSNJZH_MEDIAN', 'ABSNJZH_STDDEV',
                  'SAVNCPP_MEDIAN', 'SAVNCPP_STDDEV',
                  'USFLUX_MEDIAN', 'USFLUX_STDDEV',
                  'TOTFZ_MEDIAN', 'TOTFZ_STDDEV',
                  'MEANPOT_MEDIAN', 'MEANPOT_STDDEV',
                  'EPSZ_MEDIAN', 'EPSZ_STDDEV',
                  'MEANSHR_MEDIAN', 'MEANSHR_STDDEV',
                  'SHRGT45_MEDIAN', 'SHRGT45_STDDEV',
                  'MEANGAM_MEDIAN', 'MEANGAM_STDDEV',
                  'MEANGBT_MEDIAN', 'MEANGBT_STDDEV',
                  'MEANGBZ_MEDIAN', 'MEANGBZ_STDDEV',
                  'MEANGBH_MEDIAN', 'MEANGBH_STDDEV',
                  'MEANJZH_MEDIAN', 'MEANJZH_STDDEV',
                  'TOTFY_MEDIAN', 'TOTFY_STDDEV',
                  'MEANJZD_MEDIAN', 'MEANJZD_STDDEV',
                  'MEANALP_MEDIAN', 'MEANALP_STDDEV',
                  'TOTFX_MEDIAN', 'TOTFX_STDDEV']

    abt = pd.DataFrame(columns=abt_header)

    # Get lists of data from partition
    FL = os.listdir(partition_location + "/FL")
    NF = os.listdir(partition_location + "/NF")

    count = 0
    # Add row to abt from mvssample object and its median and std data
    for d in FL + NF:

        # Use temp list for each row and temp df
        list2add = []
        tempdf = pd.DataFrame(columns=abt_header)

        # Get mvs object and add flare type
        if d in FL:
            mvs = read_mvts_instance(partition_location + '/FL', d)
        else:
            mvs = read_mvts_instance(partition_location + '/NF', d)
        list2add.append(mvs.get_flare_type())

        # Set up temp df for future concat with master data frame object
        tempdf2 = calculate_descriptive_features(mvs.get_data())
        templist = tempdf2.to_numpy()

        # From data frame concat current with temp for each feature
        for i in templist[0]:
            list2add.append(i)
            continue
        tempdf.loc[45] = list2add
        abt = pd.concat([abt, tempdf], ignore_index=True, axis=0)

        ''' Limit to 10000 files for testing'''
        # count +=1
        # if count >= 10000:
        #     break
        # continue

    # return the completed analitics base table
    return abt



def main() -> None:
    df = process_partition(partition1_dir, "partition1")
    df.to_csv(f"{other_directory}parition1_abt.csv")
 #    params = ['TOTUSJH',
 # 'TOTBSQ',
 # 'TOTPOT',
 # 'TOTUSJZ',
 # 'ABSNJZH',
 # 'SAVNCPP',
 # 'USFLUX',
 # 'TOTFZ',
 # 'MEANPOT',
 # 'EPSZ',
 # 'MEANSHR',
 # 'SHRGT45',
 # 'MEANGAM',
 # 'MEANGBT',
 # 'MEANGBZ',
 # 'MEANGBH',
 # 'MEANJZH',
 # 'TOTFY',
 # 'MEANJZD',
 # 'MEANALP',
 # 'TOTFX',
 # 'R_VALUE']



if __name__ == "__main__":
    main()
