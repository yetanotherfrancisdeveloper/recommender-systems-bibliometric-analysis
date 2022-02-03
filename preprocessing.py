import collections
import nltk
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm

stopwords = nltk.corpus.stopwords.words('english')


def add_word_count(df: pd.DataFrame, col_name: str):
    df['word_count'] = df[col_name].apply(lambda x: len(str(x).split(' ')))

    return df


def get_pos_tag(words_list):
    tagged_words = pos_tag(words_list)

    return tagged_words


def corpus_words_frequency(corpus):
    corpus_words = [word for text in corpus for word in text.split(' ')]
    corpus_words_count = collections.Counter(corpus_words)

    return corpus_words_count


def get_words_for_pos(words_list: list, tagged_words: list, pos: str):
    words_to_get = [word for word in tqdm(words_list) if (word, pos) in tagged_words]

    return words_to_get


def clean_corpus_by_pos(corpus: list, pos_words: list):
    new_corpus = [' '.join([word for word in pos_words if word.lower() in text.lower()]) for text in tqdm(corpus)]

    return new_corpus


def remove_most_frequent_words(corpus: list, words_count: dict, top_k=0.1):
    words_count_sorted = sorted(words_count.items(), key=lambda x: x[1], reverse=True)
    words_count_sorted_dict = dict(words_count_sorted)
    # Words to remove
    words_to_remove = list(words_count_sorted_dict.keys())[:top_k]
    print(f'\nTop {top_k} most frequent words: ')
    print(words_to_remove)

    new_corpus = [text.replace(word, '') for word in words_to_remove for text in tqdm(corpus)]

    return new_corpus


def pre_processing(df: pd.DataFrame, col_name: str):
    df_len = len(df)

    corpus = []
    for i in tqdm(range(df_len)):
        # Get string from df and remove punctuation
        text = re.sub('[^a-zA-Z]', ' ', df[col_name][i])

        # Remove tags
        text = re.sub('&lt;/?.*?&gt;', ',', text)

        # Remove special characters and digits
        text = re.sub('(\\d|\\W)+', ' ', text)

        # Convert to list from string
        split_text = text.split()

        # Lemmatisation
        lemmatizer = WordNetLemmatizer()
        split_lem_text = [lemmatizer.lemmatize(word) for word in split_text if word not in stopwords]
        text = ' '.join(split_lem_text)
        corpus.append(text)

    return corpus


def plot_words_distribution(words_count: dict):
    df = pd.DataFrame({'words': list(words_count.keys()), 'count': list(words_count.values())})
    bins = np.histogram(df['count'], bins=40)[1]

    plt.hist(df['count'], bins, alpha=.8, edgecolor='red', density=True, label='Count')
    df['count'].plot.kde()
    plt.xlim(left=0)
    plt.axvline(df['count'].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(df['count'].median(), color='gray', linestyle='dashed', linewidth=1)
    plt.show()
    plt.close()
