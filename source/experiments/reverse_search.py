import pandas as pd
from matplotlib.colors import ListedColormap

from source.common_imports import *
from source.constants import *
from distfit import distfit
from scipy.stats import ks_2samp, chisquare, chi2_contingency, relfreq, kstest
import scipy.stats as stats
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


def plot_histograms(sinha_df, flare_df):
    mx_df = flare_df.loc[flare_df["AR_class"] == 1]
    nb_df = flare_df.loc[flare_df["AR_class"] == 0]

    sinha_df1 = sinha_df.loc[sinha_df["AR_class"] == 1]
    sinha_df2 = sinha_df.loc[sinha_df["AR_class"] == 0]
    num_bins = 22
    for column in SINHA_PARAMETERS:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

        mx_min = min(sinha_df1[column].min(), mx_df[column].min())
        mx_max = max(sinha_df1[column].max(), mx_df[column].max())
        nb_min = min(sinha_df2[column].min(), nb_df[column].min())
        nb_max = max(sinha_df2[column].max(), nb_df[column].max())
        mx_range = (mx_min, mx_max)
        nb_range = (nb_min, nb_max)

        density = False

        y = ax[0, 0].hist(sinha_df1[column], bins=num_bins, range=mx_range,
                          density=density,
                          weights=np.zeros_like(sinha_df1[column]) + 1. / sinha_df1[column].size)
        ax[0, 0].set_title(f"Sinha MX {column}, bins={num_bins}")
        ax[0, 0].set_xlabel("Unit")
        ax[0, 0].set_ylabel("Relative Frequency")

        y = ax[0, 1].hist(mx_df[column], bins=num_bins, range=mx_range,
                      density=density,
                          weights=np.zeros_like(mx_df[column]) + 1. / mx_df[column].size)
        ax[0, 1].set_title(f"Singh MX {column}, bins={num_bins}")
        ax[0, 1].set_xlabel("Unit")
        ax[0, 1].set_ylabel("Relative Frequency")

        y = ax[1, 0].hist(sinha_df2[column], bins=num_bins, range=nb_range,
                      density=density,
                          weights=np.zeros_like(sinha_df2[column]) + 1. / sinha_df2[column].size)
        ax[1, 0].set_title(f"Sinha NAB {column}, bins={num_bins}")
        ax[1, 0].set_xlabel("Unit")
        ax[1, 0].set_ylabel("Relative Frequency")


        y = ax[1, 1].hist(nb_df[column], bins=num_bins, range=nb_range,
                      density=density,
                          weights=np.zeros_like(nb_df[column]) + 1. / nb_df[column].size)
        ax[1, 1].set_title(f"Singh NAB {column}, bins={num_bins}")
        ax[1, 1].set_xlabel("Unit")
        ax[1, 1].set_ylabel("Relative Frequency")

        fig.tight_layout()
        # dir = "no_n/"
        dir = ""
        fig.savefig(f"{FIGURE_DIRECTORY}{dir}{column.lower()}_distribution.png")
        fig.show()

        # print(y)
        # exit(1)

        # print(y[0])
        # print(y[1])
        # print(np.multiply(y[0], np.diff(y[1])))
        # print(np.sum(y[0] * np.diff(y[1])))
        # exit(1)

    #     stat, p = ks_2samp(X1, X2)
    #     p_values.append(p)
    # x2_df.loc[len(x2_df)] = p_values


def plot_coincidence_histograms(flare_df):
    fig, ax = plt.subplots(nrows=10, ncols=3, figsize=(20, 28))
    min_max_ranges = {}
    nb_df = flare_df.loc[flare_df["AR_class"] == 0]
    mx_df = flare_df.loc[flare_df["AR_class"] == 1]
    for df, flare_class in zip([nb_df, mx_df], ["nb", "mx"]):
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
                              range=min_max_ranges[col], legend=True, alpha=0.5)
                ax[row_index, col_index].set_title(f"Singh {coincidence.capitalize()} Flares {col}")
                ax[row_index, col_index].set_xlabel("Unit")
                ax[row_index, col_index].set_ylabel("Frequency")
                ax[row_index, col_index].legend(["NB", "MX"])
    fig.suptitle(f"NB/MX 24h Timepoint Before Flare Peak Time Coincidence Histograms", y=0.99, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIGURE_DIRECTORY}nb_mx_peak_timepoint_coincidence_histograms.png")
    fig.show()
    exit(1)


def plot_coincidence_histograms2(flare_df):
    nb_df = flare_df.loc[flare_df["AR_class"] == 0]
    mx_df = flare_df.loc[flare_df["AR_class"] == 1]

    num_bins = 22

    coin_nb_df = nb_df.loc[nb_df["COINCIDENCE"] == True]
    noncoin_nb_df = nb_df.loc[nb_df["COINCIDENCE"] != True]
    coin_mx_df = mx_df.loc[mx_df["COINCIDENCE"] == True]
    noncoin_mx_df = mx_df.loc[mx_df["COINCIDENCE"] != True]

    for row_index, parameter in enumerate(SINHA_PARAMETERS):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        min_max_range = [nb_df[parameter].min(), nb_df[parameter].max()]
        coin_nb_df.hist(column=parameter, ax=ax[0], bins=num_bins, density=True,
                              range=min_max_range)
        noncoin_nb_df.hist(column=parameter, ax=ax[1], bins=num_bins, density=True,
                        range=min_max_range)
        ax[0].set_title(f"Singh NB Coincident {parameter}")
        ax[0].set_xlabel("Unit")
        ax[0].set_ylabel("Relative Frequency")
        ax[1].set_title(f"Singh NB Non-coincident {parameter}")
        ax[1].set_xlabel("Unit")
        ax[1].set_ylabel("Relative Frequency")
        ax[1].set_ylim(ax[0].get_ylim())
        fig.suptitle(f"Singh NB {parameter} Flare Coincidence Histograms",
                     y=0.99, fontweight="bold")
        fig.savefig(f"{FIGURE_DIRECTORY}nb_coincidence_histogram_{parameter.lower()}.png")
        plt.show()

    for row_index, parameter in enumerate(SINHA_PARAMETERS):
        fig, ax = plt.subplots(nrows=1, ncols=2)
        min_max_range = [nb_df[parameter].min(), nb_df[parameter].max()]
        coin_mx_df.hist(column=parameter, ax=ax[0], bins=num_bins, density=True,
                              range=min_max_range)
        noncoin_mx_df.hist(column=parameter, ax=ax[1], bins=num_bins, density=True,
                        range=min_max_range)
        ax[0].set_title(f"Coincident")
        ax[0].set_xlabel("Unit")
        ax[0].set_ylabel("Relative Frequency")
        ax[1].set_title(f"Non-coincident")
        ax[1].set_xlabel("Unit")
        ax[1].set_ylabel("Relative Frequency")
        ax[1].set_ylim(ax[0].get_ylim())
        fig.suptitle(f"Singh MX {parameter} Flare Coincidence Histograms", y=0.99, fontweight="bold")
        fig.savefig(f"{FIGURE_DIRECTORY}mx_coincidence_histogram_{parameter.lower()}.png")
        plt.show()

    # for row_index, parameter in enumerate(SINHA_PARAMETERS):
    #     min_max_range = [nb_df[parameter].min(), nb_df[parameter].max()]
    #     coin_nb_df.hist(column=parameter, ax=ax[row_index, 1], bins=num_bins,
    #                           range=min_max_range, legend=True, alpha=0.5)
    #     noncoin_nb_df.hist(column=parameter, ax=ax[row_index, 1], bins=num_bins,
    #                     range=min_max_range, legend=True, alpha=0.5)
    #     ax[row_index, 1].set_title(f"NB {parameter}")
    #     ax[row_index, 1].set_xlabel("Unit")
    #     ax[row_index, 1].set_ylabel("Frequency")
    #     ax[row_index, 1].legend(["Coincident", "Non-coincident"])
    #
    # for row_index, parameter in enumerate(SINHA_PARAMETERS):
    #     min_max_range = [nb_df[parameter].min(), nb_df[parameter].max()]
    #     coin_mx_df.hist(column=parameter, ax=ax[row_index, 0], bins=num_bins,
    #                           range=min_max_range, legend=True, alpha=0.5)
    #     noncoin_mx_df.hist(column=parameter, ax=ax[row_index, 0], bins=num_bins,
    #                     range=min_max_range, legend=True, alpha=0.5)
    #     ax[row_index, 0].set_title(f"MX {parameter}")
    #     ax[row_index, 0].set_xlabel("Unit")
    #     ax[row_index, 0].set_ylabel("Frequency")
    #     ax[row_index, 0].legend(["Coincident", "Non-coincident"])


    # fig.tight_layout()
    # fig.savefig(f"{FIGURE_DIRECTORY}nb_mx_peak_timepoint_coincidence_histograms2.png")
    # fig.show()
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
    df2.index = ["MX", "NAB"]
    cmap = ListedColormap(['red', 'green'])
    sns.heatmap(df2, cmap=cmap, annot=df.values, square=True, cbar=False, fmt=".2f")
    plt.title("Two-Sample Kolmogorov-Smirnov Tests on MX/NAB Flares,\n"
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
    df.index = SINHA_PARAMETERS
    df.to_csv(OTHER_DIRECTORY + "non_mx_means_stds.csv")
    print(df)
    exit(1)


def chi_square_test(sinha_df_, flare_df):
    mx_df = flare_df.loc[flare_df["AR_class"] == 1]
    nb_df = flare_df.loc[flare_df["AR_class"] == 0]

    sinha_df1 = sinha_df_.loc[sinha_df_["AR_class"] == 1]
    sinha_df2 = sinha_df_.loc[sinha_df_["AR_class"] == 0]

    num_bins = 5

    for sinha_df, singh_df, flare_class in zip([sinha_df1, sinha_df2], [mx_df, nb_df], ["mx", "nb"]):
        chisq_stats = []
        p_values = []
        reject_values = []

        sinha_n = sinha_df.shape[0]
        singh_n = singh_df.shape[0]
        for param in SINHA_PARAMETERS:
            minimum = min(sinha_df[param].min(), singh_df[param].min())
            maximum = max(sinha_df[param].max(), singh_df[param].max())
            min_max_range = (minimum, maximum)
            singh_freq, _, _, _ = relfreq(singh_df[param], numbins=num_bins,
                                          defaultreallimits=min_max_range)
            sinha_freq, _, _, _ = relfreq(sinha_df[param], numbins=num_bins,
                                          defaultreallimits=min_max_range)
            print(param)
            chisq, p = chisquare(singh_freq, sinha_freq)
            print(chisq)
            print(p)
            chisq_stats.append(chisq)
            p_values.append(p)
            reject_values.append(p < 0.05)
        chi_2_df = pd.DataFrame({
            "chisq_stat": chisq_stats,
            "p_value": p_values,
            "reject_95_conf": reject_values,
        })
        chi_2_df.index = SINHA_PARAMETERS
        chi_2_df.to_csv(f"{OTHER_DIRECTORY}{flare_class}_chisquare_test.csv")


def ks_test(sinha_df_, flare_df):
    mx_df = flare_df.loc[flare_df["AR_class"] == 1]
    nb_df = flare_df.loc[flare_df["AR_class"] == 0]

    sinha_df1 = sinha_df_.loc[sinha_df_["AR_class"] == 1]
    sinha_df2 = sinha_df_.loc[sinha_df_["AR_class"] == 0]

    alpha = 0.05
    num_bins = 30

    for sinha_df, singh_df, flare_class in zip([sinha_df1, sinha_df2], [mx_df, nb_df], ["mx", "nb"]):
        ks_stats = []
        p_values = []
        reject_values = []

        for param in SINHA_PARAMETERS:
            minimum = min(sinha_df[param].min(), singh_df[param].min())
            maximum = max(sinha_df[param].max(), singh_df[param].max())
            min_max_range = (minimum, maximum)
            num_bins = int(np.sqrt(min(sinha_df[param].size, singh_df[param].size)))
            singh_freq, _, _, _ = relfreq(singh_df[param], numbins=num_bins,
                                          defaultreallimits=min_max_range)
            sinha_freq, _, _, _ = relfreq(sinha_df[param], numbins=num_bins,
                                          defaultreallimits=min_max_range)
            stat, p_value = kstest(singh_freq, sinha_freq)
            ks_stats.append(stat)
            p_values.append(p_value)
            reject_values.append(p_value < alpha)

        ks_df = pd.DataFrame({
            "ks_stat": ks_stats,
            "p_value": p_values,
            "reject_95_conf": reject_values,
        })
        ks_df.index = SINHA_PARAMETERS
        print(ks_df)
        ks_df.to_csv(f"{OTHER_DIRECTORY}{flare_class}_ks_test.csv")


def main():
    sinha_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}sinha_dataset.csv")
    sinha_df1 = sinha_df.loc[sinha_df["AR_class"] == 1]
    sinha_df2 = sinha_df.loc[sinha_df["AR_class"] == 0]
    flare_df = pd.read_csv(f"{FLARE_DATA_DIRECTORY}singh_nbmx_data.csv")


    # flare_df = flare_df.loc[flare_df["xray_class"] != "N"]
    mx_df = flare_df.loc[flare_df["AR_class"] == 1]
    nb_df = flare_df.loc[flare_df["AR_class"] == 0]
    classes = ["N", "A", "B", "M", "X"]
    for coincidence in [True, False]:
        df = flare_df.loc[flare_df["COINCIDENCE"] == coincidence]
        for c in classes:
            coin_df = df.loc[df["xray_class"] == c]
            print(coin_df.shape[0], end=" ")
        print()
    for c in classes:
        df = flare_df.loc[flare_df["xray_class"] == c]
        print(df.shape[0], end=" ")


    exit(1)

    plot_coincidence_histograms2(flare_df)
    # chi_square_test(sinha_df, flare_df)
    # ks_test(sinha_df, flare_df)
    # plot_histograms(sinha_df, flare_df)
    # x2_df.index = ["MX", "NON_MX"]
    # x2_df.to_csv(OTHER_DIRECTORY + "two_sample_ks.csv")
    # compute_mean_std(nb_df, sinha_df2)
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