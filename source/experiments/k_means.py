################################################################################
# Filename: k_means.py
# Description: Todo
################################################################################

# Custom Imports
from copy import copy

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "k_means"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    lo_time = 5
    hi_time = 17
    flare_classes = ["NB", "MX"]
    colors = {
        "N": "gray",
        "B": "dodgerblue",
        "C": "green",
        "M": "orange",
        "X": "red",
        "NB": "dodgerblue",
        "MX": "orangered",
    }

    markers = {
        "N": "^",
        "B": "v",
        "C": "s",
        "M": "<",
        "X": ">",
        "NB": ".",
        "MX": "x",
    }

    flare_class_caption = "/".join(flare_classes)
    flare_class_filename = "_".join(flare_classes)

    flare_dataframes = [
        get_ar_properties(flare_class, lo_time, hi_time,
                          coincidence_flare_classes="nbmx").dropna()
        for flare_class in flare_classes
    ]

    all_flares_df = pd.concat(flare_dataframes)
    # all_flares_df["xray_class"].replace("M", "MX", inplace=True)
    # all_flares_df["xray_class"].replace("X", "MX", inplace=True)
    # all_flares_df["xray_class"].replace("N", "NB", inplace=True)
    # all_flares_df["xray_class"].replace("B", "NB", inplace=True)
    # all_flares_df["xray_class"].replace("C", "BC", inplace=True)

    file = f"{RESULTS_DIRECTORY}correlation/other/" \
           f"nb_mx_anova_f_scores_{get_time_window(lo_time, hi_time)}.csv"
    anova_df = pd.read_csv(file)
    params = anova_df.iloc[:5]["parameter"].values
    n = params.shape[0]

    X = all_flares_df[params].to_numpy()

    # X = StandardScaler().fit_transform(X)
    X = MinMaxScaler().fit_transform(X)
    y = all_flares_df["xray_class"]

    pca = PCA(n_components=n, random_state=10)
    lda = LinearDiscriminantAnalysis()
    principal_components = pca.fit_transform(X)  # Plot the explained variances
    linear_discriminants = lda.fit_transform(X, y)
    features_num = range(pca.n_components_)
    ev = sum(pca.explained_variance_ratio_[:3]) * 100
    pca_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in features_num])


    # Uncomment this for LDA plot.
    # lda_df = pd.DataFrame(linear_discriminants, columns=[f"LD{i + 1}" for i in range(len(flare_classes) - 1)])
    # lda_df["xray_class"] = list(y)
    # for flare_class in ["NB", "C", "MX"]:
    #     flare_lda_df = lda_df.loc[lda_df["xray_class"] == flare_class]
    #     flare_lda_df.plot(kind="scatter", x="LD1", y="LD2",
    #                       ax=ax, color=colors[flare_class],
    #                       alpha=0.5,
    #                       label=flare_class)
    # plt.title("NB/C/MX Flares, 5h-17h, LDA")

    pca_df["xray_class"] = list(y)
    pca_df["magnitude"] = list(all_flares_df["magnitude"])
    # pca_df = pca_df.loc[pca_df["xray_class"] != "N"]
    # pca_df = pca_df.loc[pca_df["xray_class"] != "B"]
    # for flare_class in flare_classes:
    #     for single_flare_class in flare_class:
    #         flare_pca_df = pca_df.loc[pca_df["xray_class"] == single_flare_class]
    #         flare_pca_df.plot(kind="scatter", x="PC1", y="PC2",
    #                           ax=ax[0], color=colors[single_flare_class],
    #                           alpha=0.5,
    #                           label=single_flare_class)
    # plt.legend()
    # plt.title(f"{flare_class_caption} Flares, {get_time_window(lo_time, hi_time)}, PCA,\n"
    #           f"Explained Variance of {ev:.2f}%")
    # plt.show()

    for flare_class in flare_classes:
        for single_flare_class in flare_class:
            all_flares_df["xray_class"].replace(single_flare_class, flare_class, inplace=True)


    parameters = [2, 3, 4, 5, 6]
    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})
    best_score = -1
    kmeans_model = KMeans(random_state=10)  # instantiating KMeans model
    silhouette_scores = []

    plt.clf()
    cluster_colors = ["blue", "orange", "green", "red"]
    for param, param_num in zip(parameter_grid, parameters):
        fig, ax = plt.subplots(2, figsize=(19, 11))
        for flare_class in flare_classes:
            for single_flare_class in flare_class:
                flare_pca_df = pca_df.loc[pca_df["xray_class"] == single_flare_class]
                flare_pca_df.plot(kind="scatter", x="PC1", y="PC2",
                                  ax=ax[0], color=colors[single_flare_class],
                                  alpha=0.5,
                                  label=single_flare_class)
        ax[0].set_title(f"{flare_class_caption} Flares, {get_time_window(lo_time, hi_time)}, PCA K-Means,\n"
                        f"k={param_num}, Explained Variance of {ev:.2f}%, Original Data")
        kmeans_model.set_params(**param)
        clusters = kmeans_model.fit_predict(pca_df.iloc[:, :3])
        centroids = (kmeans_model.cluster_centers_)
        pca_df["clusters"] = [f"Cluster {label}" for label in clusters]
        fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3",
                            color="clusters",
                            symbol="xray_class",
                            size="magnitude",
                            opacity=0.8,
                            title=f"{flare_class_caption} Flares, "
                                  f"{get_time_window(lo_time, hi_time)}, "
                                  f"PCA {param_num}-Means, "
                                  f"Explained Variance of {ev:.2f}%, "
                                  f"Using Top {n} ANOVA Parameters",
                            symbol_sequence=["circle", "square", "diamond-open", "x"])
        fig.write_html(
            f"{other_directory}{flare_class_filename.lower()}_{get_time_window(lo_time, hi_time)}_{param_num}_means_pca_3d_top_{n}_anova.html")

        # for k_index in range(param_num):
        #     label_df = pca_df[clusters == k_index]
        #     ax[1].scatter(label_df.iloc[:, 0], label_df.iloc[:, 1], alpha=0.5)
        #     ax[1].scatter(centroids[:, 0], centroids[:, 1], color="k")
        #     ax[1].set_xlabel("PCA1")
        #     ax[1].set_ylabel("PCA2")
        #     ax[1].set_title(f"K-Means Clustering, k={param_num}")
        # if param_num == 2:
        #     y_true = pca_df["xray_class"].replace("N", "NB").replace("B", "NB") \
        #         .replace("M", "MX").replace("X", "MX").to_numpy()
        #     flare_classes_reverse = copy(flare_classes)
        #     flare_classes_reverse.reverse()
        #     print(flare_classes)
        #     print(flare_classes_reverse)
        #     y_pred = np.array([flare_classes_reverse[i] for i in clusters])
        #     write_classification_metrics(
        #         y_true, y_pred,
        #         filename=f"{metrics_directory}{flare_class_filename.lower()}_{get_time_window(lo_time, hi_time)}_{param_num}_means_pca_2d.txt",
        #         clf_name="K Means, K=2",
        #         flare_classes=flare_classes_reverse
        #     )

        # plotting the results
        # plt.tight_layout()
        # plt.savefig(f"{figure_directory}{flare_class_filename.lower()}_{param_num}_means_clustering_minmax_{get_time_window(lo_time, hi_time)}.png")
        # plt.show()
    # evaluation based on silhouette_score
    # for p in parameter_grid:
    #     kmeans_model.set_params(**p)  # set current hyper parameter
    #     kmeans_model.fit(X)  # fit model on wine dataset, this will find clusters based on parameter p
    #     ss = metrics.silhouette_score(X, kmeans_model.labels_)  # calculate silhouette_score
    #     silhouette_scores += [ss]  # store all the scores
    #     print('Parameter:', p, 'Score', ss)
    #     # check p which has the best score
    #     if ss > best_score:
    #         best_score = ss
    #         best_grid = p
    # # plotting silhouette score
    # plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', width=0.5)
    # plt.xticks(range(len(silhouette_scores)), list(parameters))
    # plt.title(f'{flare_class_caption} Flares, {get_time_window(lo_time, hi_time)}, K-Means Clustering, Silhouette Score')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel("Silhouette Score")
    # plt.tight_layout()
    # plt.savefig(figure_directory + f"{flare_class_filename.lower()}_silhoutte_scores_minmax_{get_time_window(lo_time, hi_time)}.png")
    # plt.show()
    # plt.clf()

    # distortions = []
    # inertias = []
    # k_values = [i for i in range(2, 11)]
    # for k in k_values:
    #     kmeans_model = KMeans(k)
    #     kmeans_model.fit(X)
    #     inertias.append(kmeans_model.inertia_)
    #     distortions.append(sum(np.min(cdist(X, kmeans_model.cluster_centers_,
    #                                         'euclidean'), axis=1)) / X.shape[0])
    # plt.plot(k_values, distortions)
    # plt.xlabel("K")
    # plt.ylabel("Distortion")
    # plt.title(f"{flare_class_caption} Flares, {get_time_window(lo_time, hi_time)}, K-Means Clustering,\n"
    #           f"Distortion Elbow Method\n"
    #           f"(Using Average of Squared Euclidean Distance Between Centroids)")
    # plt.tight_layout()
    # plt.savefig(figure_directory + f"{flare_class_filename.lower()}_distortion_elbow_minmax_{get_time_window(lo_time, hi_time)}.png")
    # plt.show()
    # plt.clf()
    #
    # plt.plot(k_values, inertias)
    # plt.xlabel('K')
    # plt.ylabel('Inertia')
    # plt.title(f"{flare_class_caption} Flares, {get_time_window(lo_time, hi_time)}, K-Means Clustering,\n"
    #           f"Inertia Elbow Method\n"
    #           f"(Using Sum of Squared Distances to Closest Centroid")
    # plt.tight_layout()
    # plt.savefig(figure_directory + f"{flare_class_filename.lower()}_inertia_elbow_minmax_{get_time_window(lo_time, hi_time)}.png")
    # plt.show()


if __name__ == "__main__":
    main()
