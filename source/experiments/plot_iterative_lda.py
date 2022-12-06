iters = [i for i in range(0, 11)]

from matplotlib import pyplot as plt
import numpy as np

def main():
    figures = r"C:\Users\youar\PycharmProjects\flare_forecasting\results\lda_classifier\figures/"
    metrics = r"C:\Users\youar\PycharmProjects\flare_forecasting\results\lda_classifier\metrics/"
    def get_filename(trim, coincidence, iter):
        if trim:
            return f"{metrics}{coincidence}/nb_mx_loo_0h_24h_{trim}_trimmed_means_{coincidence}_{iter}.txt"
        else:
            return f"{metrics}{coincidence}/nb_mx_loo_0h_24h_trimmed_means_{coincidence}_{iter}.txt"

    for coincidence in ["all", "coincident", "noncoincident"]:
        # for trim in [None, "15", "17"]:
        tss = []
        for i in range(11):
            with open(get_filename(None, coincidence, i), "r") as fp:
                lines = fp.readlines()
                tss.append(float(lines[-1].split("TSS: ")[1]))

        plt.clf()
        plt.plot(tss, label="LDA")
        plt.xticks(iters)
        plt.xlabel("Iterations of LDA")
        plt.ylabel("True Skill Score (TSS)")
        plt.title(f"NB/MX {coincidence.capitalize()} Flares, 0h-24h, Iterative LDA,\n"
                  f"Trimmed Means, George Method")
        plt.axvline(np.argmax(tss), c="k", ls="--", label="Maximum")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{figures}george_{coincidence}_iterative_lda_summary.png")
        plt.show()

if __name__ == "__main__":
    main()