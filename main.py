import argparse
import json
import nltk
import numpy as np
import os
import pandas as pd
from clustering import (dbscan_clustering, generate_keywords, get_keywords_list,
                        get_sentence_embeddings, lda_clustering)
from config import DATA_PATH, POS_PATH, TXT_PATH
from plotter import bar_plot, plot_publications_series, plot_words_distribution
from preprocessing import (add_word_count, clean_corpus_by_pos, compute_tf_idf,
                           corpus_words_frequency, get_pos_tag, get_words_for_pos,
                           pre_processing, remove_most_frequent_words, remove_short_texts, remove_short_words)
from tqdm import tqdm
from utils import from_json_to_list_tuples, get_pdf_from_txt_name, obj_to_file, read_data, save_json

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
pd.set_option('display.max_columns', None)


def main(clustering_model, skip_data_viz):
    if not skip_data_viz:
        df = read_data()
        print('Dataframe columns:\n')
        for col in df.columns:
            print(f'- {col}')

        print('\nFirst 5 rows of dataframe: ')
        print(df.head(5))

        # Adding word count column
        df = add_word_count(df, 'summary')
        # Pre-process abstracts
        print('\nPre-processing abstracts ...')
        corpus = pre_processing(df, 'summary')

        # Get words' frequency
        print('\nGetting words\' frequency ...')
        words_count = corpus_words_frequency(corpus)
        words_list = list(words_count.keys())
        # Get tags for unique words
        print('\nGetting words for POS tag ...')
        tagged_words = get_pos_tag(words_list)
        nn_words = get_words_for_pos(tagged_words, 'NN')
        nns_words = get_words_for_pos(tagged_words, 'NNS')
        nnp_words = get_words_for_pos(tagged_words, 'NNP')

        # Concatenating all lists of nouns
        noun_words = nn_words + nns_words + nnp_words

        print('\nGetting words to plot ...')
        words_to_plot = {k: v for k, v in words_count.items() if k in noun_words}
        sorted_words_to_plot = sorted(words_to_plot.items(), key=lambda x: x[1], reverse=True)
        words_to_plot = dict(sorted_words_to_plot)
        # Bar plot of top k words
        bar_plot(list(words_to_plot.keys()), list(words_to_plot.values()))
        # Bar plot of publications per year
        plot_publications_series(df, remove_years=np.array(['2022']))

    if clustering_model == 'keybert':
        df = read_data()
        if not os.path.exists(os.path.join(DATA_PATH, 'keywords.txt')):
            print('\nFinding keywords with KeyBert ...')
            docs = df['summary']
            keywords = generate_keywords(docs)
            obj_to_file(keywords, 'keywords')
        else:
            print('\nKeybert - Collecting keywords from existing .txt file ...')
            with open(os.path.join(DATA_PATH, 'keywords.txt')) as file:
                keywords = json.load(file)

        keywords_list = get_keywords_list(keywords, mode='best')
        obj_to_file(keywords_list, 'best_keywords')
        df['keywords'] = keywords_list
        print('\nSaving to .csv ...')
        df.to_csv(os.path.join(DATA_PATH, 'df_keybert.csv'))

    elif clustering_model == 'dbscan':
        df = read_data()
        print('\nDBSCAN clustering ...')
        if not os.path.exists(os.path.join(DATA_PATH, 'embeddings.npy')):
            summaries = df['summary']
            embeddings_ndarray = get_sentence_embeddings(summaries)
            np.save(os.path.join(DATA_PATH, 'embeddings.npy'), embeddings_ndarray)
        else:
            embeddings_ndarray = np.load(os.path.join(DATA_PATH, 'embeddings.npy'))

        print('\nSaving to .csv ...')
        dbscan_clustering(df, embeddings_ndarray)

    elif clustering_model == 'lda':
        # Clustering with the Latent Dirichlet Allocation (LDA) algorithm
        txt_list = os.listdir(TXT_PATH)
        txt_str_list = []

        print('\nOpening .txt files for the analysis ...')
        for txt in tqdm(txt_list):
            with open(os.path.join(TXT_PATH, txt), encoding='utf-8') as f:
                txt_str_list.append(f.read())

        txt_df = pd.DataFrame({'article': txt_str_list, 'file': txt_str_list})
        print('\nPre-processing articles for LDA ...')
        corpus = pre_processing(txt_df, 'article')

        # Removing short texts
        print('\nRemoving short texts ...')
        corpus, short_texts_idxes = remove_short_texts(corpus)

        # Remove texts from
        try:
            for idx in range(len(short_texts_idxes)):
                txt_list.pop(short_texts_idxes[idx] - idx)
        except (IndexError, TypeError) as e:
            print(f'{e}: check the threshold or the list!')

        # Removing short words
        print('\nRemoving short words ...')
        corpus = remove_short_words(corpus)

        print('\nGetting words\' frequency ...')
        words_count = corpus_words_frequency(corpus)
        words_list = list(words_count.keys())

        print('\nGetting words for POS tag ...')
        if os.path.exists(POS_PATH):
            tagged_words = from_json_to_list_tuples(json_path=POS_PATH)
            nn_words = get_words_for_pos(tagged_words, 'NN')
        else:
            tagged_words = get_pos_tag(words_list)
            save_json(out_path=POS_PATH, data=tagged_words)
            nn_words = get_words_for_pos(tagged_words, 'NN')

        # Concatenating all lists of nouns
        noun_words = nn_words

        # Getting new corpus by removing non POS words
        print('\nGetting new corpus with only POS tag words ...')
        new_corpus = clean_corpus_by_pos(corpus, noun_words)
        print('\nGetting words\' frequency ...')
        new_words_count = corpus_words_frequency(new_corpus)
        plot_words_distribution(new_words_count)

        # Remove most frequent words
        new_corpus_clean = remove_most_frequent_words(new_corpus, new_words_count)

        # LDA clustering with new corpus
        txt_df = pd.DataFrame({'article': new_corpus_clean})
        results_df = lda_clustering(txt_df, column='article')
        # Computing top 20 words by tf-idf
        print('\nComputing tf-idf ...')
        top_tf_idf_words, tf_idf_results = compute_tf_idf(results_df)
        # Printing top 20 words by tf-idf
        topics = results_df.topics.unique()
        topics.sort()
        for topic in topics:
            print(f"\nTF-IDF TOP {20} WORDS FOR TOPIC #{topic}")
            print(top_tf_idf_words[topic])

        # Adding .txt and .pdf files' names to csv and saving it
        print('\nSaving csv with results ...')
        results_df['file'] = txt_list
        results_df = get_pdf_from_txt_name(results_df)
        results_df.to_csv(f'{DATA_PATH}/lda_results.csv')


if __name__ == '__main__':
    # Adding parsing of arguments for terminal
    parser = argparse.ArgumentParser(description='Clustering papers on recommender systems into topics')
    parser.add_argument('-skip_data_viz',
                        action='store_true',
                        help='Creates and saves plots on the papers data. '
                             'Value to use: True | False')
    parser.add_argument('-clustering_model',
                        type=str,
                        default='lda',
                        help='Model to use in order to cluster data. '
                             'Values to use: "lda" | "dbscan" | "keybert"')
    args = parser.parse_args()
    main(args.clustering_model,
         args.skip_data_viz)
