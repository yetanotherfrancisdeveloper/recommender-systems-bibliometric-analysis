import os
import json
import numpy as np
from keybert import KeyBERT
from matplotlib import colors as m_colors
from matplotlib import pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from utils import obj_to_file, read_data, DATA_PATH

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


def lda_clustering(df, n_topics=5, min_df=3, max_df=0.95):
    # Create Count Vectorizer instance and Document X Term matrix (dtm)
    print('\nCreating Count Vectorizer and Document X Term matrix ...')
    cv = CountVectorizer(max_df=max_df, min_df=min_df, stop_words="english")
    dtm = cv.fit_transform(df['summary'])

    # Creat the model
    print('\nCreating LDA model ...')
    lda_model = LatentDirichletAllocation(n_components=n_topics,
                                          max_iter=30, random_state=2022)
    # Fit model
    print('\nFitting the LDA model ...')
    lda_model.fit(dtm)

    for i, topic in enumerate(lda_model.components_):
        print("THE TOP {} WORDS FOR TOPIC #{}".format(10, i))
        print([cv.get_feature_names_out()[index] for index in topic.argsort()[-10:]])
        print("\n")

    final_topics = lda_model.transform(dtm)
    df["topics"] = final_topics.argmax(axis=1)

    my_colors = np.array([color for name, color in m_colors.TABLEAU_COLORS.items()])
    # TSNE for visualization of the clusters
    print('\nCreating and fitting TSNE to visualize clusters in 2D ...')
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(final_topics)
    print(tsne_lda)

    print('\nColors: ')
    print(my_colors)
    plt.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=my_colors[n_topics])
    plt.show()
    plt.close()


def main():
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
    # Clustering with the Latent Dirichlet Allocation (LDA) algorithm
    df.to_csv(os.path.join(DATA_PATH, 'df_extended.csv'))
    lda_clustering(df)


if __name__ == '__main__':
    main()
