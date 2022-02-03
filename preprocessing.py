import collections
import nltk
import numpy as np
import pandas as pd
import re
from config import PLOT_PATH
from matplotlib import pyplot as plt
from nltk import pos_tag
from nltk.corpus import words
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


def get_words_for_pos(tagged_words: list, pos: str):
    tagged_words_dict = dict(tagged_words)
    tagged_words_df = pd.DataFrame({'word': list(tagged_words_dict.keys()), 'pos': list(tagged_words_dict.values())})
    words_to_get = tagged_words_df.loc[tagged_words_df.pos == pos].word.tolist()

    return words_to_get


def clean_corpus_by_pos(corpus: list, pos_words: list):
    new_corpus = [' '.join([word for word in pos_words if word.lower() in text.lower()]) for text in tqdm(corpus)]

    return new_corpus


def remove_most_frequent_words(corpus: list, words_count: dict, top_k=100):
    words_count_sorted = sorted(words_count.items(), key=lambda x: x[1], reverse=True)
    words_count_sorted_dict = dict(words_count_sorted)
    # Words to remove
    words_to_remove = list(words_count_sorted_dict.keys())[:top_k]
    print(f'\nTop {top_k} most frequent words: ')
    print(words_to_remove)

    new_corpus = [' '.join([word for word in text.split() if word not in words_to_remove])
                  for text in tqdm(corpus)]

    return new_corpus


def remove_short_words(corpus: list, threshold=4):
    words_lists = [text.split() for text in corpus]
    only_long_words = [' '.join([word for word in word_list if len(word) > threshold])
                       for word_list in tqdm(words_lists)]

    return only_long_words


def remove_short_texts(corpus: list, threshold=100):
    only_long_texts = [text for text in tqdm(corpus) if len(text) > threshold]
    short_texts = [text for text in corpus if len(text) <= threshold]

    return only_long_texts, short_texts


def pre_processing(df: pd.DataFrame, col_name: str, lemmatize=False):
    df_len = len(df)

    corpus = []
    lemmatizer = WordNetLemmatizer()
    voc = set(words.words())
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
        if lemmatize:
            split_lem_text = [lemmatizer.lemmatize(word) for word in split_text
                              if word not in stopwords and word in voc]
            text = ' '.join(split_lem_text)
            corpus.append(text)
        else:
            split_lem_text = [word for word in split_text
                              if word not in stopwords and word in voc]
            text = ' '.join(split_lem_text)
            corpus.append(text)

    return corpus


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
