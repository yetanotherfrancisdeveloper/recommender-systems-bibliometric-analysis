import json
import numpy as np
import os
import pandas as pd
from config import DATA_PATH, TXT_PATH, PLOT_PATH
from keybert import KeyBERT
from matplotlib import pyplot as plt
from preprocessing import (pre_processing, corpus_words_frequency, get_words_for_pos, get_pos_tag, clean_corpus_by_pos,
                           remove_most_frequent_words, remove_short_words, remove_short_texts)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from tqdm import tqdm
from utils import obj_to_file, read_data, get_pdf_from_txt_name

keywords_model = KeyBERT()


def generate_keywords(docs):
    keywords = list()

    for idx, summary in enumerate(docs):
        if idx % 50 == 0:
            print(f'Processing paper {idx+1}')
        keywords.append(keywords_model.extract_keywords(summary, keyphrase_ngram_range=(1, 2), stop_words=None))
    return keywords


def get_keywords_list(keyword_score_list, mode = 'all'):
    keywords_list = list()
    if mode == 'all':
        for keywords_per_paper in keyword_score_list:
            keywords = list()
        for keyword_score in keywords_per_paper:
            keywords.append(keyword_score[0])
        keywords_list.append(keywords)
    elif mode == 'best':
        for keywords_per_paper in keyword_score_list:
            keywords = [keyword_score[0] for keyword_score in keywords_per_paper]
            scores = [keyword_score[1] for keyword_score in keywords_per_paper]
            best_keyword = keywords[np.argmax(scores)]
            keywords_list.append(best_keyword)
    return keywords_list


def lda_clustering(df, n_topics=5, min_df=0.05, max_df=0.95, column='summary'):
    # Create Count Vectorizer instance and Document X Term matrix (dtm)
    print('\nCreating Count Vectorizer and Document X Term matrix ...')
    cv = CountVectorizer(max_df=max_df, min_df=min_df, stop_words="english")
    dtm = cv.fit_transform(df[column])

    # Creat the model
    print('\nCreating LDA model ...')
    lda_model = LatentDirichletAllocation(n_components=n_topics,
                                          max_iter=30, random_state=2022)
    # Fit model
    print('\nFitting the LDA model ...')
    lda_model.fit(dtm)

    for i, topic in enumerate(lda_model.components_):
        print(f"THE TOP {20} WORDS FOR TOPIC #{i}")
        print([cv.get_feature_names_out()[index] for index in topic.argsort()[-20:]])
        print("\n")

    final_topics = lda_model.transform(dtm)
    df["topics"] = final_topics.argmax(axis=1)

    # TSNE for visualization of the clusters
    print('\nCreating and fitting TSNE to visualize clusters in 2D ...')
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(final_topics)
    print('\nt-SNE shape: ', tsne_lda.shape)

    for g in np.unique(df.topics):
        label_idx = np.where(df.topics == g)
        tsne_lda_label = tsne_lda[label_idx]
        plt.scatter(x=tsne_lda_label[:, 0], y=tsne_lda_label[:, 1], cmap='turbo', label=f'Topic {g}')

    plt.legend()
    plt.savefig(f'{PLOT_PATH}/t-SNE_LDA_(topics={n_topics},min_df={min_df},max_df={max_df}.png')
    plt.close()

    return df


def main(model='keybert'):
    if model == 'keybert':
        df = read_data()
        if not os.path.exists(os.path.join(DATA_PATH, 'keywords.txt')):
            docs = df['summary']
            keywords = generate_keywords(docs)
            obj_to_file(keywords, 'keywords')
        else:
            with open(os.path.join(DATA_PATH, 'keywords.txt')) as file:
                keywords = json.load(file)

        keywords_list = get_keywords_list(keywords, mode='best')
        obj_to_file(keywords_list, 'best_keywords')
        df['keywords'] = keywords_list
        df.to_csv(os.path.join(DATA_PATH, 'df_extended.csv'))
    else:
        # Clustering with the Latent Dirichlet Allocation (LDA) algorithm
        txt_list = os.listdir(TXT_PATH)
        txt_str_list = []
        for txt in tqdm(txt_list):
            with open(os.path.join(TXT_PATH, txt), encoding='utf-8') as f:
                txt_str_list.append(f.read())

        txt_df = pd.DataFrame({'article': txt_str_list, 'file': txt_str_list})
        print('\nPre-processing articles for LDA ...')
        corpus = pre_processing(txt_df, 'article')

        # Removing short texts
        print('\nRemoving short texts ...')
        corpus, short_texts = remove_short_texts(corpus)

        # Remove texts from
        for text in short_texts:
            txt_list.remove(text)

        # Removing short words
        print('\nRemoving short words ...')
        corpus = remove_short_words(corpus)

        print('\nGetting words\' frequency ...')
        words_count = corpus_words_frequency(corpus)
        words_list = list(words_count.keys())
        words_count_sorted = sorted(words_count.items(), key=lambda x: x[1], reverse=True)
        words_count_sorted_dict = dict(words_count_sorted)
        # Words to remove
        print(list(words_count_sorted_dict.keys())[:100])

        print('\nGetting words for POS tag ...')
        # TODO: save the POS tags in a .json
        tagged_words = get_pos_tag(words_list)
        nn_words = get_words_for_pos(tagged_words, 'NN')

        # Concatenating all lists of nouns
        noun_words = nn_words

        # Getting new corpus by removing non POS words
        print('\nGetting new corpus with only POS tag words ...')
        new_corpus = clean_corpus_by_pos(corpus, noun_words)
        print('\nGetting words\' frequency ...')
        new_words_count = corpus_words_frequency(new_corpus)

        # Remove most frequent words
        new_corpus_clean = remove_most_frequent_words(new_corpus, new_words_count)

        # LDA clustering with new corpus
        txt_df = pd.DataFrame({'article': new_corpus_clean})
        results_df = lda_clustering(txt_df, column='article')
        results_df['file'] = txt_list
        results_df = get_pdf_from_txt_name(results_df)
        results_df.to_csv(f'{DATA_PATH}/lda_results.csv')


if __name__ == '__main__':
    main(model='lda')
