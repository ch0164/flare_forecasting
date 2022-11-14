import pandas as pd

from source.common_imports import *
from source.constants import *


def main():
    sinha_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_dataset.csv")
    flare_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}b_data.txt", header=0, delimiter=r"\s+")

    col = "TOTUSJH"
    sinha_params = [
        "TOTPOT_y",
        "TOTUSJZ_y",
        "ABSNJZH_y",
        "SAVNCPP_y",
        "USFLUX_y",
        "AREA_ACR_y",
        "MEANPOT_y",
        "R_VALUE_y",
        "SHRGT45_y",
        "EPSZ",
        "TOTBSQ",
        "TOTFZ",
        "TOTABSTWIST"
    ]
    renamed_params = [
        "TOTPOT",
        "TOTUSJZ",
        "ABSNJZH",
        "SAVNCPP",
        "USFLUX",
        "AREA_ACR",
        "MEANPOT",
        "R_VALUE",
        "SHRGT45",
        "EPSZ",
        "TOTBSQ",
        "TOTFZ",
        "TOTABSTWIST"
    ]
    all_params = ["index", "T_REC", "NOAA_AR", "TOTUSJH"] + sinha_params

    sinha_df1 = sinha_df.loc[sinha_df["AR_class"] == 0]
    sinha_df1.reset_index(inplace=True)

    sinha_df1.sort_values(by=col, inplace=True)
    flare_df1 = flare_df.sort_values(by=col)


    print(sinha_df1["TOTUSJH"].std(), sinha_df1["ABSNJZH"].std())

    df = pd.merge_asof(sinha_df1, flare_df1, on=col, tolerance=0.1, direction="nearest")

    # df = pd.merge_asof(sinha_df1, flare_df1, left_on='TOTUSJH', right_on='SHRGT45', tolerance=2)

    print(df.columns)
    df = df[all_params].sort_values(by="index", ascending=True)
    # df = df[all_params]
    df.drop("index", axis=1, inplace=True)
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)
    df.rename(columns={k: v for k, v in zip(sinha_params, renamed_params)}, inplace=True)

    df.to_csv(f"{FLARE_DATA_DIRECTORY}sinha_b_data.csv", index=False)



if __name__ == "__main__":
    main()