import os
import numpy as np
import pandas as pd
from config import PLOT_PATH
from matplotlib import pyplot as plt


def bar_plot(words_list: list, frequencies: list, width=0.35, top_k=10):
    x = np.arange(top_k)
    plt.bar(x, frequencies[:top_k], width=width)
    plt.xticks(x, words_list[:top_k], rotation=25)

    if not os.path.isdir(PLOT_PATH):
        os.mkdir(PLOT_PATH)

    plt.savefig(os.path.join(PLOT_PATH, f'top_{top_k}_words_by_frequency.png'))
    plt.close()


def plot_publications_series(df, remove_years=np.array([])):
    # Creating column 'year'
    df['year'] = df.published.apply(lambda x: x[-4:])

    # Getting data in dictionary
    data = {year: len(df[df.year == year]) for year in df.year.unique()}

    # Remove years if given
    if remove_years:
        for year in remove_years:
            try:
                del data[year]
            except KeyError:
                print(f"The year {year} wasn't in the data! Check again.")
                continue

    # Sorting dictionary by year in ascending order
    sorted_data = sorted(data.items(), key=lambda x: x[0])
    sorted_data = dict(sorted_data)

    # Getting from and to years
    from_year = min(list(sorted_data.keys()))
    to_year = max(list(sorted_data.keys()))

    # Bar plot
    plt.bar(np.arange(len(sorted_data.keys())), list(sorted_data.values()), width=0.35)
    plt.xticks(np.arange(len(sorted_data.keys())), list(sorted_data.keys()))

    if not os.path.isdir(PLOT_PATH):
        os.mkdir(PLOT_PATH)

    plt.savefig(os.path.join(PLOT_PATH, f'bar_plot_publications_per_year_{from_year}-{to_year}.png'))
    plt.close()


def plot_words_distribution(words_count: dict):
    df = pd.DataFrame({'words': list(words_count.keys()), 'count': list(words_count.values())})
    bins = np.histogram(df['count'], bins=40)[1]
    # Histogram of words' count
    plt.hist(df['count'], bins, alpha=.8, edgecolor='red', density=True, label='Count')
    # Plot KDE
    df['count'].plot.kde()

    plt.xlim(left=0)
    # Compute and plot mean and median of distribution
    plt.axvline(df['count'].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(df['count'].median(), color='gray', linestyle='dashed', linewidth=1)

    plt.savefig(f'{PLOT_PATH}/words_distribution.png')
    plt.close()
