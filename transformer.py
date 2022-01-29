import json
from keybert import KeyBERT
from utils import read_data, DATA_PATH

keywords_model = KeyBERT()


def generate_keywords(docs):
    keywords = list()

    for idx, summary in enumerate(docs):
      if idx % 50 == 0:
        print(f'Processing paper {idx+1}')
      keywords.append(keywords_model.extract_keywords(summary, keyphrase_ngram_range=(1, 2), stop_words=None))
    return keywords


def list_to_file(a_list, file_path):
    with open(f'{DATA_PATH}/{file_path}.txt', 'w') as file:
        json.dump(a_list, file, ensure_ascii=False)


def main():
    df = read_data()
    docs = df['summary']
    keywords = generate_keywords(docs)
    list_to_file(keywords, 'keywords')


if __name__ == '__main__':
    main()
