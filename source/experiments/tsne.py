################################################################################
# Filename: experiment_template.py
# Description: This file is a correlation for easy creation of new experiments.
################################################################################

# Custom Imports
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from source.utilities import *


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Experiment Name (No Acronyms)
    experiment = "tsne"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    lo_time = 5
    hi_time = 17
    flare_classes = ["NB", "MX"]
    flare_class_caption = "_".join(flare_classes).lower()
    random_state = 7
    time_window = f"{lo_time}h_{hi_time}h"
    time_window_caption = time_window.replace("_", "-")
    mx_classfied_by = "mx_classified_by"

    # ------------------------------------------------------------------------

    flare_dataframes = [
        get_ar_properties(flare_class, lo_time, hi_time,
                          coincidence_time_window="0h_24h",
                          coincidence_flare_classes="nbmx").dropna()
        for flare_class in flare_classes
    ]
    all_flares_df = pd.concat(flare_dataframes)
    all_flares_df = all_flares_df. \
        reset_index(). \
        drop(["index"], axis=1). \
        rename_axis("index")

    all_flares_df["xray_class"].replace("M", "MX", inplace=True)
    all_flares_df["xray_class"].replace("X", "MX", inplace=True)
    all_flares_df["xray_class"].replace("N", "NB", inplace=True)
    all_flares_df["xray_class"].replace("B", "NB", inplace=True)

    file = f"{RESULTS_DIRECTORY}correlation/other/" \
           f"nb_mx_anova_f_scores_{get_time_window(lo_time, hi_time)}.csv"
    anova_df = pd.read_csv(file)
    params = anova_df.iloc[:5]["parameter"].values
    n = params.shape[0]

    X = all_flares_df[params].to_numpy()
    # X = all_flares_df[FLARE_PROPERTIES]
    X = MinMaxScaler().fit_transform(X)
    y = all_flares_df["xray_class"].to_numpy()

    X_embedded = TSNE(n_components=2, random_state=1).fit_transform(X)

    loo = LeaveOneOut()
    loo.get_n_splits(X_embedded)

    for k in [2, 3, 4, 5, 6]:
        clf = KNeighborsClassifier(k)
        y_true = []
        y_pred = []
        for train_index, test_index in loo.split(X_embedded):
            X_train, X_test = X_embedded[train_index], X_embedded[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)[0]
            truth = y_test[0]
            y_pred.append(pred)
            y_true.append(truth)

        # Write classification metrics of NB/MX.
        filename = f"{metrics_directory}/top_5_anova/knn_{k}_" \
                   f"{flare_class_caption}_{time_window}_classification_metrics.txt"
        write_classification_metrics(y_true, y_pred, filename, f"KNN {k}",
                                     flare_classes=flare_classes,
                                     print_output=False)


    # classes = ["NB", "MX"]
    # markers_list = [".", "x"]
    # colors = ["orangered", "dodgerblue"]
    # markers = {c: m for c, m in zip(classes, markers_list)}
    # clusters = {i: colors[i] for i in range(2)}
    # for k in [2]:
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     X_embedded = TSNE(n_components=2, random_state=1).fit_transform(X)
    #     model = KMeans(n_clusters=k, init="k-means++", random_state=1, tol=1e-7)
    #     labels = model.fit_predict(X_embedded)
    #     centers = np.array(model.cluster_centers_)
    #
    #     print(labels)
    #
    #     plt.scatter(X_embedded[1046, 0], X_embedded[1046, 1], label="NB", marker=".", color="k")
    #     plt.scatter(X_embedded[1047, 0], X_embedded[1047, 1], label="MX", marker="x", color="k")
    #     for index, (x, label, pred) in enumerate(zip(X_embedded, y, labels)):
    #         plt.scatter(x[0], x[1], marker=markers[label], color=clusters[pred])
    #     plt.scatter(centers[:, 0], centers[:, 1], marker="d", color='k')
    #     # This is done to find the centroid for each clusters.
    #     plt.title(f"NB/MX Flares, {time_window_caption}, TSNE K-Means Clustering, K=2,\n"
    #               f"All Parameters")
    #     plt.legend()
    #     plt.savefig(f"{figure_directory}nb_mx_5h_17h_kmeans_all_parameters.png")
    #     plt.show()


if __name__ == "__main__":
    main()
