import nltk
import pandas
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
pd.set_option('display.max_columns', None)
stopwords = nltk.corpus.stopwords.words('english')


def read_data():
    papers_df = pd.read_csv('data/df.csv')
    papers_df.drop('Unnamed: 0', axis=1, inplace=True)

    return papers_df


def add_word_count(df: pandas.DataFrame, col_name: str):
    df['word_count'] = df[col_name].apply(lambda x: len(str(x).split(' ')))

    return df


def pre_processing(df: pandas.DataFrame, col_name: str):
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

        # Stemming
        # stemmer = PorterStemmer()

        # Lemmatisation
        lemmatizer = WordNetLemmatizer()
        split_lem_text = [lemmatizer.lemmatize(word) for word in split_text if word not in stopwords]
        text = ' '.join(split_lem_text)
        corpus.append(text)

    return corpus


def main():
    df = read_data()
    print('Dataframe columns:\n')
    for col in df.columns:
        print(f'- {col}')
    # Adding word count column
    df = add_word_count(df, 'summary')
    # Pre-process abstracts
    print('\nPre-processing abstracts ...')
    corpus = pre_processing(df, 'summary')


if __name__ == '__main__':
    main()
