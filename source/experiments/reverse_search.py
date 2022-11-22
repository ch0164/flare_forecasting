import pandas as pd
from matplotlib.colors import ListedColormap

from source.common_imports import *
from source.constants import *
from distfit import distfit
from scipy.stats import ks_2samp
import numpy as np

SINHA_PARAMETERS = [
    "TOTUSJH",
    "USFLUX",
    "TOTUSJZ",
    "R_VALUE",
    "TOTPOT",
    "AREA_ACR",
    "SAVNCPP",
    "ABSNJZH",
    "MEANPOT",
    "SHRGT45",
]
EXPERIMENT_DIRECTORY = r"C:\Users\youar\PycharmProjects\flare_forecasting\results\sinha_singh_distribution_comparison/"
FIGURE_DIRECTORY = f"{EXPERIMENT_DIRECTORY}figures/"
OTHER_DIRECTORY = f"{EXPERIMENT_DIRECTORY}other/"

x2_df = pd.DataFrame(columns=SINHA_PARAMETERS)


def plot_histograms(df1, df2, label=None):
    p_values = []
    for column in SINHA_PARAMETERS:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        df1.hist(column=column, ax=ax[0, 0], bins=7)
        ax[0, 0].set_title(f"Sinha {label.upper()} {column}")
        ax[0, 0].set_xlabel("Unit")
        ax[0, 0].set_ylabel("Frequency")
        X1 = df1[column].dropna()
        dist = distfit(smooth=5)
        dist.fit_transform(X1, verbose=False)
        dist.plot(ax=ax[0, 1])

        df2.hist(column=column, ax=ax[1, 0], bins=7)
        ax[1, 0].set_title(f"Singh {label.upper()} {column}")
        ax[1, 0].set_xlabel("Unit")
        ax[1, 0].set_ylabel("Frequency")
        X2 = df2[column].dropna()
        dist = distfit(smooth=5)
        dist.fit_transform(X2, verbose=False)
        dist.plot(ax=ax[1, 1])

        fig.tight_layout()
        fig.savefig(f"{FIGURE_DIRECTORY}{label}_{column.lower()}_distribution.png")
        fig.show()

    #     stat, p = ks_2samp(X1, X2)
    #     p_values.append(p)
    # x2_df.loc[len(x2_df)] = p_values


def plot_coincidence_histograms(df, flare_class):
    fig, ax = plt.subplots(nrows=10, ncols=3, figsize=(20, 28))
    min_max_ranges = {}
    for col_index, coincidence in enumerate(COINCIDENCES):
        flare_df = df.copy()
        if coincidence == "coincident":
            flare_df = flare_df.loc[flare_df["COINCIDENCE"] == True]
        elif coincidence == "noncoincident":
            flare_df = flare_df.loc[flare_df["COINCIDENCE"] == False]
        else:
            for col in SINHA_PARAMETERS:
                min_max_ranges[col] = [flare_df[col].min(), flare_df[col].max()]
        for row_index, col in enumerate(SINHA_PARAMETERS):
            flare_df.hist(column=col, ax=ax[row_index, col_index], bins=7,
                          range=min_max_ranges[col])
            ax[row_index, col_index].set_title(f"Singh {coincidence.upper()} {col}")
            ax[row_index, col_index].set_xlabel("Unit")
            ax[row_index, col_index].set_ylabel("Frequency")
    fig.suptitle(f"Singh {flare_class} Histograms", y=0.99, fontweight="bold")
    fig.tight_layout()
    fig.show()
    exit(1)


def plot_null_hypothesis(df):
    df2 = pd.DataFrame(columns=SINHA_PARAMETERS)
    def accept_reject(p):
        alpha = 0.05
        print(p)
        if p >= alpha:
            return 1
        else:
            return 0
    for col in SINHA_PARAMETERS:
        df2[col] = df[col].apply(accept_reject)
    df2.index = ["MX", "Non-MX"]
    cmap = ListedColormap(['red', 'green'])
    sns.heatmap(df2, cmap=cmap, annot=df.values, square=True, cbar=False, fmt=".2f")
    plt.title("Two-Sample Kolmogorov-Smirnov Tests on MX/Non-MX Flares,\n"
              "Confidence Level = 0.05")
    plt.show()
    exit(1)


def get_median_df(cols):
    df = pd.DataFrame()
    for col in cols:
        temp_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_mx_data_{col.lower()}.csv")
        df[col] = temp_df["T_REC"]
    df = df.T
    date_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_dataset.csv")
    date_df = date_df.loc[date_df["AR_class"] == 1]
    dates = []
    for i in range(df.shape[1]):
        dates.append(df[i].mode(dropna=True)[0])
    date_df.insert(loc=0, column="T_REC", value=dates)
    print(date_df)
    date_df.to_csv(f"{FLARE_DATA_DIRECTORY}sinha_mx_data_majority.csv", index=False)
    return date_df


def get_flare_info(df):
    info_df = pd.read_csv(f"{FLARE_LIST_DIRECTORY}mx_list.txt")
    info_df["time_peak"] = pd.to_datetime(info_df["time_peak"])
    df["T_REC"] = df["T_REC"].str.replace(".000_TAI", "")
    df["T_REC"] = df["T_REC"].str.replace(".", "-")
    df["T_REC"] = df["T_REC"].str.replace("_", " ")
    df["T_REC"] = pd.to_datetime(df["T_REC"])
    df["time_peak"] = df["T_REC"] + pd.Timedelta(days=1)
    df.sort_values(by="time_peak", inplace=True)
    info_df.sort_values(by="time_peak", inplace=True)
    recorded_times_df = pd.merge_asof(df, info_df, by="time_peak", direction="nearest", tolerance=pd.Timedelta(hours=24))
    recorded_times_df.to_csv(f"{FLARE_LIST_DIRECTORY}sinha_flare_times.csv", index=False)


def compute_mean_std(df1, df2):
    singh_means = [df1[col].mean() for col in SINHA_PARAMETERS]
    sinha_means = [df2[col].mean() for col in SINHA_PARAMETERS]
    singh_stds = [df1[col].std() for col in SINHA_PARAMETERS]
    sinha_stds = [df2[col].std() for col in SINHA_PARAMETERS]
    d = {
            "singh_means": singh_means,
            "sinha_means": sinha_means,
            "singh_stds": singh_stds,
            "sinha_stds": sinha_stds,
    }
    df = pd.DataFrame(d)
    df.to_csv(OTHER_DIRECTORY + "non_mx_means_stds.csv")
    print(df)
    exit(1)




def main():
    sinha_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_dataset.csv")
    sinha_df1 = sinha_df.loc[sinha_df["AR_class"] == 1]
    flare_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}mx_data.txt", header=0, delimiter=r"\s+")
    mx_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}time_series_means/"
            f"original_coincidence_definition/0h_24h/mx_0h_24h_mean_dataset.csv")
    # plot_histograms(sinha_df1, mx_df, "mx")

    sinha_df2 = sinha_df.loc[sinha_df["AR_class"] == 0]
    nab_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}time_series_means/"
                        f"original_coincidence_definition/0h_24h/nb_0h_24h_nbmx_mean_dataset.csv")
    # plot_histograms(sinha_df2, nab_df, "non_mx_only_b")
    # x2_df.index = ["MX", "NON_MX"]
    # x2_df.to_csv(OTHER_DIRECTORY + "two_sample_ks.csv")
    # plot_coincidence_histograms(nab_df, "NB")
    compute_mean_std(nab_df, sinha_df2)
    exit(1)
    cols = ["TOTUSJH", "TOTUSJZ", "SAVNCPP", "R_VALUE", "SHRGT45", "ABSNJZH", "TOTPOT", "AREA_ACR", "USFLUX"]
    for col in cols:
        sinha_params = [
            "MEANPOT_y",
            "EPSZ",
            "TOTBSQ",
            "TOTFZ",
            "TOTABSTWIST"
        ]
        renamed_params = [
            "MEANPOT",
            "EPSZ",
            "TOTBSQ",
            "TOTFZ",
            "TOTABSTWIST"
        ]
        all_params = ["index", "T_REC", "NOAA_AR", col] + sinha_params

        sinha_df1 = sinha_df.loc[sinha_df["AR_class"] == 1]
        sinha_df1.reset_index(inplace=True)

        sinha_df1.sort_values(by=col, inplace=True)
        flare_df1 = flare_df.sort_values(by=col)

        df = pd.merge_asof(sinha_df1, flare_df1, on=col, tolerance=0.01 * sinha_df1[col].std(), direction="nearest")

        # df = pd.merge_asof(sinha_df1, flare_df1, left_on='TOTUSJH', right_on='SHRGT45', tolerance=2)

        print(df.columns)
        df = df[all_params].sort_values(by="index", ascending=True)
        # df = df[all_params]
        df.drop("index", axis=1, inplace=True)
        df.reset_index(inplace=True)
        df.drop("index", axis=1, inplace=True)
        df.rename(columns={k: v for k, v in zip(sinha_params, renamed_params)}, inplace=True)
        print(df.head(100).to_string())
        df.to_csv(f"{FLARE_DATA_DIRECTORY}sinha_mx_data_{col.lower()}.csv", index=False)

    df = get_median_df(cols)
    get_flare_info(df)




if __name__ == "__main__":
    main()