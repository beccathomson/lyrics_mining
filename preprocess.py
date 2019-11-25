from sys import exit
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
stop_words = set(w.lower() for w in stopwords.words('english'))
english_vocab = set(w.lower() for w in words.words())


def main():
    df = pd.read_csv("./raw_lyrics.csv", header=0, index_col=0)
    print('Loaded file into dataframe')
    df.dropna(inplace=True)
    print('Dropped rows with missing values')
    df['lyrics'] = df['lyrics'].map(lambda lyric: ' '.join([tok.lower() for tok in word_tokenize(lyric) if
                                                    tok.isalpha() and tok.lower() not in stop_words and tok.lower() in english_vocab]))
    print('Cleaned lyrics')
    df.to_csv('./clean_lyrics.csv', sep=',', encoding='utf-8', index=False)
    print('Wrote to csv file')


if __name__ == '__main__':
    main()
    exit()
