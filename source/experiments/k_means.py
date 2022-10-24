################################################################################
# Filename: k_means.py
# Description: Todo
################################################################################

# Custom Imports
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
    flare_classes = ["C", "MX"]
    flare_class_caption = "/".join(flare_classes)
    flare_class_filename = "_".join(flare_classes)

    flare_dataframes = [
        get_ar_properties(flare_class, lo_time, hi_time,
                          coincidence_flare_classes="nbcmx").dropna()
        for flare_class in flare_classes
    ]

    all_flares_df = pd.concat(flare_dataframes)
    all_flares_df["xray_class"].replace("M", "MX", inplace=True)
    all_flares_df["xray_class"].replace("X", "MX", inplace=True)
    # all_flares_df["xray_class"].replace("N", "NBC", inplace=True)
    # all_flares_df["xray_class"].replace("B", "BC", inplace=True)
    # all_flares_df["xray_class"].replace("C", "BC", inplace=True)

    X = all_flares_df[FLARE_PROPERTIES].to_numpy()
    X = StandardScaler().fit_transform(X)
    y = all_flares_df["xray_class"]

    dataset_all = pd.DataFrame(X, columns=FLARE_PROPERTIES)
    pca_2 = PCA(n_components=2)
    pca_2.fit_transform(X)
    pca_3 = PCA(n_components=3)
    pca_3.fit_transform(X)
    dataset_pca_2 = pd.DataFrame(abs(pca_2.components_), columns=FLARE_PROPERTIES, index=['PC1', 'PC2'])
    dataset_pca_3 = pd.DataFrame(abs(pca_3.components_), columns=FLARE_PROPERTIES, index=['PC1', 'PC2', 'PC3'])

    parameters = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]
    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})
    best_score = -1
    kmeans_model = KMeans()  # instantiating KMeans model
    silhouette_scores = []
    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)  # set current hyper parameter
        kmeans_model.fit(X)  # fit model on wine dataset, this will find clusters based on parameter p
        ss = metrics.silhouette_score(X, kmeans_model.labels_)  # calculate silhouette_score
        silhouette_scores += [ss]  # store all the scores
        print('Parameter:', p, 'Score', ss)
        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p
    # plotting silhouette score
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title(f'{flare_class_caption} Flares, {get_time_window(lo_time, hi_time)}, K-Means Clustering, Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(figure_directory + f"{flare_class_filename.lower()}_silhoutte_scores_{get_time_window(lo_time, hi_time)}.png")
    plt.show()
    plt.clf()

    distortions = []
    inertias = []
    k_values = [i for i in range(2, 11)]
    for k in k_values:
        kmeans_model = KMeans(k)
        kmeans_model.fit(X)
        inertias.append(kmeans_model.inertia_)
        distortions.append(sum(np.min(cdist(X, kmeans_model.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
    plt.plot(k_values, distortions)
    plt.xlabel("K")
    plt.ylabel("Distortion")
    plt.title(f"{flare_class_caption} Flares, {get_time_window(lo_time, hi_time)}, K-Means Clustering,\n"
              f"Distortion Elbow Method\n"
              f"(Using Average of Squared Euclidean Distance Between Centroids)")
    plt.tight_layout()
    plt.savefig(figure_directory + f"{flare_class_filename.lower()}_distortion_elbow_{get_time_window(lo_time, hi_time)}.png")
    plt.show()
    plt.clf()

    plt.plot(k_values, inertias)
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.title(f"{flare_class_caption} Flares, {get_time_window(lo_time, hi_time)}, K-Means Clustering,\n"
              f"Inertia Elbow Method\n"
              f"(Using Sum of Squared Distances to Closest Centroid")
    plt.tight_layout()
    plt.savefig(figure_directory + f"{flare_class_filename.lower()}_inertia_elbow_{get_time_window(lo_time, hi_time)}.png")
    plt.show()


if __name__ == "__main__":
    main()
