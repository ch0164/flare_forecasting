tss = [
0.7849750730617157,
0.7979928638675939,
0.8105483429457512,
0.8025023932151362,
0.7973995047429173,
0.7941281181022002,
0.7771145341339725,
0.7713549948971115,
0.7510700242881668,
0.7280951589806881,
]

iters = [i for i in range(0, 10)]

from matplotlib import pyplot as plt
import numpy as np

def main():
    plt.plot(tss, label="LDA")
    plt.xticks(iters)
    plt.xlabel("Iterations of LDA")
    plt.ylabel("True Skill Score (TSS)")
    plt.title("NB/MX Flares, 0h-24h, Iterative LDA")
    plt.axvline(np.argmax(tss), c="k", ls="--", label="Maximum")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()