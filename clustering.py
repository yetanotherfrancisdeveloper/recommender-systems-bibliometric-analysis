import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from utils import DATA_PATH
from sentence_transformers import SentenceTransformer


def get_sentence_embeddings(sentences, model = 'all-MiniLM-L6-v2'):
    sbert_model = SentenceTransformer(model)
    embeddings = sbert_model.encode(sentences)
    return embeddings

def main():
    df = pd.read_csv(os.path.join(DATA_PATH, 'df.csv'))
    if not os.path.exists(os.path.join(DATA_PATH, 'embeddings.npy')):
        summaries = df['summary']
        embeddings_ndarray = get_sentence_embeddings(summaries)
        np.save(os.path.join(DATA_PATH, 'embeddings.npy'), embeddings_ndarray)
    else:
        embeddings_ndarray = np.load(os.path.join(DATA_PATH, 'embeddings.npy'))
    clustering = DBSCAN(eps=0.5, min_samples=500, metric='cosine').fit(embeddings_ndarray)
    clusters = clustering.labels_
    print('Cluster labels', np.unique(clusters))
    df['cluster'] = clusters
    print(df['cluster'].value_counts())
    df.to_csv(os.path.join(DATA_PATH, 'df.csv'))


if __name__ == '__main__':
    main()