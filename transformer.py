import os
import json
import numpy as np
from keybert import KeyBERT
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
    df.to_csv(os.path.join(DATA_PATH, 'df.csv'))


if __name__ == '__main__':
    main()
