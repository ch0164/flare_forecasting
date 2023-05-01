from source.constants import *
import datetime
import pandas as pd
from datetime import datetime as dt_obj

def parse_tai_string(tstr: str):
    if "not applicable" in tstr:
        return "not applicable"
    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])
    return dt_obj(year, month, day, hour, minute)


flare_list = pd.read_csv(f"{FLARE_LIST_DIRECTORY}nbcmx_list.csv", header=0, parse_dates=["time_start", "time_peak", "time_end"], index_col="index")

x_list = flare_list.loc[flare_list["xray_class"] == "X"]
m_list = flare_list.loc[flare_list["xray_class"] == "M"]
b_list = flare_list.loc[flare_list["xray_class"] == "B"]
c_list = flare_list.loc[flare_list["xray_class"] == "C"]
n_list = flare_list.loc[flare_list["xray_class"] == "N"]

mx_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_MX_ARs_and_errors.txt", header=0, delimiter=r"\s+")
mx_data["T_REC"] = mx_data["T_REC"].apply(parse_tai_string)
mx_data.set_index("T_REC", inplace=True)

b_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_OnlyB_ARs_and_errors.txt", header=0, delimiter=r"\s+")
b_data["T_REC"] = b_data["T_REC"].apply(parse_tai_string)
b_data.set_index("T_REC", inplace=True)

bc_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}Data_ABC_ARs_and_errors.txt", header=0, delimiter=r"\s+")
bc_data["T_REC"] = bc_data["T_REC"].apply(parse_tai_string)
bc_data.set_index("T_REC", inplace=True)

# bc_data = pd.read_csv(f"{FLARE_DATA_DIRECTORY}bc_data.txt", header=0, delimiter=r"\s+")
# bc_data["T_REC"] = bc_data["T_REC"].apply(parse_tai_string)
# bc_data.set_index("T_REC", inplace=True)

flare_datas = [b_data, bc_data, mx_data, mx_data]
flare_lists = [b_list, c_list, m_list, x_list]
flare_classes = ["B", "C", "M", "X"]

def floor_minute(time, cadence=12):
    if not isinstance(time, str):
        return time - datetime.timedelta(minutes=time.minute % cadence)
    else:
        return "not applicable"


if __name__ == "__main__":
    new_df = pd.DataFrame(columns=["time_start", "time_end", "time_peak", "xray_class", "nar", "COINCIDENCE"] + FLARE_PROPERTIES)
    print(new_df)
    i = 0
    for data, class_list, c in zip(flare_datas, flare_lists, flare_classes):
        for index, flare in class_list.iterrows():
            nar = flare["nar"]
            coin = flare["COINCIDENCE"]
            dt = floor_minute(pd.to_datetime(flare["time_peak"]) - datetime.timedelta(hours=24))

            flares_in_ar = data.loc[data["NOAA_AR"] == nar]
            try:
                record_index = flares_in_ar.index[flares_in_ar.index.get_loc(dt)]
            except KeyError:
                # print("KeyError")
                continue
            # time_difference = abs(record_index - dt)
            # try:
            record = flares_in_ar.loc[record_index]
            r = {"time_start": flare["time_start"],
                 "time_end": flare["time_end"],
                "time_peak": flare["time_peak"],
                 "xray_class": c,
                 "nar": nar,
                 "COINCIDENCE": coin}

            r2 = {prop: record[prop] for prop in FLARE_PROPERTIES}
            r.update(r2)
            new_row = pd.Series(r)
            new_df = pd.concat([new_df, new_row.to_frame().T])
        new_df.to_csv(f"{FLARE_DATA_DIRECTORY}new_agu_data.csv")