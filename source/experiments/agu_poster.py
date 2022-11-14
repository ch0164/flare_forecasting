from source.common_imports import *
from source.constants import *
from source.utilities import *

def main():
    flare_classes = ["B", "C", "M", "X"]
    lo_time = 0
    hi_time = 24
    flare_dataframes = [
        get_ar_properties(flare_class,
                          lo_time,
                          hi_time,
                          filter_multiple_ars=False,
                          )
        for flare_class in ["B", "C", "M", "X"]
    ]
    b_df, c_df, mx_df = tuple(flare_dataframes)
    m_df = mx_df.loc[mx_df["xray_class"] == "M"]
    x_df = mx_df.loc[mx_df["xray_class"] == "X"]
    c_df = c_df.loc[c_df["xray_class"] == "C"]
    flare_dataframes = [b_df, c_df, m_df, x_df]
    flares_df = pd.concat(flare_dataframes).dropna()
    flares_df.reset_index(inplace=True)
    flares_df.drop("index", axis=1, inplace=True)
    print(flares_df["xray_class"])



if __name__ == "__main__":
    main()