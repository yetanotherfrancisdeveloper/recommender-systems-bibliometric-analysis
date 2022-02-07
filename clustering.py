import numpy as np
import os
from config import DATA_PATH, PLOT_PATH
from keybert import KeyBERT
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE


def get_sentence_embeddings(sentences, model='all-MiniLM-L6-v2'):
    sbert_model = SentenceTransformer(model)
    embeddings = sbert_model.encode(sentences)

    return embeddings


def dbscan_clustering(df, embeddings_ndarray):
    clustering = DBSCAN(eps=0.5, min_samples=500, metric='cosine').fit(embeddings_ndarray)
    clusters = clustering.labels_
    print('Cluster labels', np.unique(clusters))
    df['cluster'] = clusters
    print(df['cluster'].value_counts())
    df.to_csv(os.path.join(DATA_PATH, 'df_dbscan_results.csv'))


def generate_keywords(docs):
    keywords_model = KeyBERT()
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
            for keyword_score in keywords_per_paper:    # I think it was meant to be nested like this?
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
    lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=30, random_state=2022)
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
