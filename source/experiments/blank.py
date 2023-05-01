import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    # create example dataframe
    data = {'Subset 1': [1000, 50, 700, 1300, 300],
            'Subset 2': [5, 7, 6, 10, 2],
            'Subset 3': [12, 18, 20, 5, 8]}
    data = {"N": [1000, 800, 500],
            "B": [1300, 1000, 800],
            "C": [300, 200, 100],
            "M": [700, 600, 500],
            "X": [50, 30, 20]}
    # index = ['N', 'B', 'C', 'M', 'X']
    index = ["Subset1", "Subset2", "Subset3"]
    df = pd.DataFrame(data, index=index)

    # create grouped bar chart
    ax = df.plot(kind='bar', figsize=(8, 6), width=0.8, stacked=True)

    # add numerical count labels to each individual bar
    for container in ax.containers:
        ax.bar_label(container, label_type='center', fontsize=10)

    # add labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('Counts')
    ax.set_title('Grouped Bar Chart of Class Counts by Subset')

    # add legend
    ax.legend(title='Subset')

    # show the plot
    plt.show()