################################################################################
# Filename: lda_classifier2.py
# Description: This file classifies flares using LDA.
################################################################################
from copy import copy

import pandas as pd

# Custom Imports
from source.utilities import *
from scipy import stats
from num2words import num2words


def main() -> None:
    # Disable Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

    # Experiment Name (No Acronyms)
    experiment = "lda_classifier2"
    experiment_caption = experiment.title().replace("_", " ")

    # ------------------------------------------------------------------------
    # Place any results in the directory for today for the current experiment.
    now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)

    # ------------------------------------------------------------------------
    # Pre-process the data for the experiment.
    # Afterwards, place this data in `cleaned_data_directory`.
    time_interval = 24
    lo_time = 0
    hi_time = lo_time + time_interval

    # Get the time window of the experiment for metadata.
    # lo_time = int((center_time - time_interval // 2))
    # hi_time = int((center_time + time_interval // 2))
    time_window = f"{lo_time}h_{hi_time}h"
    time_window_caption = time_window.replace("_", "-")

    # Obtain the properties for flares.
    flare_dataframes = [
        get_ar_properties(flare_class,
                          lo_time,
                          hi_time,
                          coincidence_time_window="0h_24h",
                          coincidence_flare_classes="nbmx"
        )
        for flare_class in ["NB", "MX"]
    ]

    nb_df, mx_df = tuple(flare_dataframes)
    nb_df.dropna(inplace=True)
    mx_df.dropna(inplace=True)
    mx_df["xray_class"] = "MX"
    nb_df["xray_class"] = "NB"

    flares_list = [nb_df, mx_df]

    for coincidence in ["coincident", "noncoincident", "all"]:
        if coincidence == "coincident":
            is_coincident = True
        elif coincidence == "noncoincident":
            is_coincident = False
        else:
            is_coincident = None
        all_flares_df = pd.concat(flares_list)
        if is_coincident is not None:
            all_flares_df = all_flares_df.loc[all_flares_df["COINCIDENCE"] == is_coincident]
        all_flares_df = all_flares_df.\
            reset_index().\
            drop(["index"], axis=1).\
            rename_axis("index")

        # Apply trimmed means.
        nb_centroid = None
        mx_centroid = None
        trimming_ratio = 0.15
        saved_df = all_flares_df
        for lda_index in range(0, 1):
            all_flares_df = saved_df.copy()
            print(f"{coincidence} LOO LDA, Iteration {lda_index}")
            test_df = pd.DataFrame()

            train_lda = LinearDiscriminantAnalysis()
            train_components = train_lda.fit_transform(
                all_flares_df[FLARE_PROPERTIES],
                all_flares_df["xray_class"])
            train_lda_df = pd.DataFrame(train_components, columns=["LD1"])
            train_lda_df.index = all_flares_df.index
            train_lda_df["xray_class"] = pd.Series(all_flares_df["xray_class"])
            nb_df = train_lda_df.loc[
                train_lda_df["xray_class"] == "NB"]
            mx_df = train_lda_df.loc[
                train_lda_df["xray_class"] == "MX"]

            # After the first iteration of LDA, do trimmed means.
            if lda_index != 0:
                if nb_centroid is None:
                    nb_centroid = nb_df.mean()
                if mx_centroid is None:
                    mx_centroid = mx_df.mean()
                # Drop 10% of the records the furthest away from the means.
                # Save them for testing.

                nb_diff = (abs(nb_df - nb_centroid))["LD1"].sort_values()
                mx_diff = (abs(mx_df - mx_centroid))["LD1"].sort_values()
                nb_to_drop = int(np.ceil(trimming_ratio * nb_diff.shape[0]))
                mx_to_drop = int(np.ceil(trimming_ratio * mx_diff.shape[0]))
                test_df = pd.concat(
                    [test_df, nb_df.drop(nb_diff.head(nb_diff.shape[0] - nb_to_drop).index)])
                test_df = pd.concat(
                    [test_df,
                     mx_df.drop(mx_diff.head(mx_diff.shape[0] - mx_to_drop).index)])
                nb_df.drop(nb_diff.tail(nb_to_drop).index, inplace=True)
                mx_df.drop(mx_diff.tail(mx_to_drop).index, inplace=True)

                # Recompute the means for the next iteration
                # based on those records that were not dropped.
                nb_centroid = nb_df.mean()
                mx_centroid = mx_df.mean()

            print(test_df.shape[0])
            # Do LOO testing on the non-trimmed records.
            if lda_index != 0:
                X = all_flares_df[FLARE_PROPERTIES].drop(test_df.index, axis=0)
                y = all_flares_df.drop(test_df.index, axis=0)["xray_class"].to_numpy()
            else:
                X = all_flares_df[FLARE_PROPERTIES]
                y = all_flares_df["xray_class"].to_numpy()
            loo = LeaveOneOut()
            loo.get_n_splits(X)
            y_true, y_pred = [], []
            y_trimmed_truths = pd.DataFrame()
            y_trimmed_predictions = pd.DataFrame()
            plot_instance = True
            for train_index, test_index in loo.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                train_lda = LinearDiscriminantAnalysis()
                train_components = train_lda.fit_transform(
                    X_train[FLARE_PROPERTIES], y_train)
                train_lda_df = pd.DataFrame(train_components, columns=["LD1"])
                train_lda_df["xray_class"] = pd.Series(y_train)

                pred = train_lda.predict(X_test[FLARE_PROPERTIES])[0]
                y_true.append(y_test[0])
                y_pred.append(pred)

                if plot_instance:
                    fig, ax = plt.subplots()
                    plot_instance = False


                    train_lda_df["jitter"] = [random.uniform(0, 1) for _ in range(train_lda_df.shape[0])]
                    for name, color in zip(["NB", "MX"], ["dodgerblue", "orangered"]):
                        df = train_lda_df.loc[train_lda_df["xray_class"] == name]
                        df.plot(x="LD1", y="jitter", label=f"{name} Train", kind="scatter", c=color, ax=ax)
                    if lda_index != 0:
                        ax.scatter(nb_centroid, 0.5, c="k", marker="+", label="NB Centroid")
                        ax.scatter(mx_centroid, 0.5, c="k", marker="x", label="MX Centroid")
                    ax.legend(loc="lower right")
                    ax.set_title(f"LDA Classifier, Trimmed Means Iteration {lda_index}, LOO,\n"
                                 f"NB/MX {coincidence.capitalize()} Flares, {time_window_caption}")
                    plt.savefig(f"{figure_directory}{coincidence}/nb_mx_lda_{int(trimming_ratio * 100)}_trimmed_means_{time_window}_{lda_index}.png")
                    plt.show()



                # Maintain a DataFrames for predictions for the trimmed records.
                if lda_index != 0:
                    pred = pd.DataFrame(train_lda.predict(
                        all_flares_df[FLARE_PROPERTIES].iloc[test_df.index]),
                                        columns=test_index)
                    y_trimmed_predictions = pd.concat([y_trimmed_predictions, pred], axis=1)

            # Add the trimmed records into the testing set.
            if lda_index != 0:
                for i in range(y_trimmed_predictions.shape[0]):
                    pred_label = y_trimmed_predictions.T[i].mode().values[0]
                    true_label = y[i]
                    y_true.append(true_label)
                    y_pred.append(pred_label)

            cm = confusion_matrix(y_true, y_pred, labels=["NB", "MX"])
            tp, fn, fp, tn = cm.ravel()
            cr = classification_report(y_true, y_pred, output_dict=True)
            accuracy = cr["accuracy"]
            sens = tp / float(tp + fn)
            far = fp / float(fp + tn)
            tss = sens - far
            custom_cr = {
                "NB": cr["NB"],
                "MX": cr["MX"],
            }
            custom_cr["NB"]["count"] = custom_cr["NB"].pop("support")
            custom_cr["MX"]["count"] = custom_cr["MX"].pop("support")
            cr_df = pd.DataFrame(custom_cr).transpose()
            with open(
                    f"{metrics_directory}{coincidence}/nb_mx_loo_{time_window}_{int(trimming_ratio * 100)}_trimmed_means_{coincidence}_{lda_index}.txt",
                    "w") as f:
                stdout = sys.stdout
                sys.stdout = f
                print("Confusion Matrix")
                print("-" * 50)
                print("  NB", " MX")
                print("NB", cm[0])
                print("MX ", cm[1])
                print("-" * 50)
                print("Classification Metrics")
                print("-" * 50)
                print(cr_df)
                print()
                print(f"Accuracy: {accuracy}")
                print(f"TSS: {tss}")
                sys.stdout = stdout


if __name__ == "__main__":
    main()
