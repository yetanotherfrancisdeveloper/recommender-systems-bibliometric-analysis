import collections
import nltk
import numpy as np
import pandas as pd
import re
from nltk import pos_tag
from nltk.corpus import words
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
    short_texts_idxes = [corpus.index(text) for text in corpus if len(text) <= threshold]

    return only_long_texts, short_texts_idxes


def compute_tf_idf(df, top_k=20):
    tfidf = TfidfVectorizer()
    topics = df.topics.unique()
    topics_docs = [df.loc[df.topics == topic].article.tolist() for topic in topics]
    # For eventual strange results there is a check for the text to be a string
    topics_docs = [[doc if isinstance(doc, str) else '' for doc in topic_corpus] for topic_corpus in topics_docs]

    top_tf_idf_words_per_topic = []
    tf_idf_results = []
    for topic_corpus in topics_docs:
        tf_idf_topic = tfidf.fit_transform(topic_corpus)
        tf_idf_results.append(tf_idf_topic)

        # Finding top words
        # Sorting vocabulary of words for which the tf-idf was computed
        sorted_vocabulary = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])
        sorted_vocabulary_dict = dict(sorted_vocabulary)
        # Getting words sorted by index in array
        vocab_array = np.array(list(sorted_vocabulary_dict.keys()))
        # Reshaping the vocabulary to the same shape of tf-idf array computed (tf_idf_topic)
        vocab_array_resh = np.tile(vocab_array, (tf_idf_topic.toarray().shape[0], 1))
        # Flattening it
        vocab_array_resh_flat = vocab_array_resh.ravel()
        # Getting indexes of array sorted by value (from lowest to highest)
        tf_idf_values_sorted = np.argsort(tf_idf_topic.toarray().ravel())
        # Getting top_k words by tf-idf
        top_tf_idf_values = tf_idf_values_sorted[-top_k:][::-1]
        top_tf_idf_words = vocab_array_resh_flat[top_tf_idf_values]
        top_tf_idf_words_per_topic.append(top_tf_idf_words.tolist())

    return top_tf_idf_words_per_topic, tf_idf_results


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
