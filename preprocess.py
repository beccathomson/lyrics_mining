from sys import exit

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import word_tokenize

stop_words = set(w.lower() for w in stopwords.words('english'))
english_vocab = set(w.lower() for w in words.words())


def stem_lyrics(song):
    return [ps.stem(word) for word in song.split()]


def main():
    global ps
    ps = nltk.stem.PorterStemmer()
    df = pd.read_csv("./raw_lyrics.csv", header=0, index_col=0)
    print('Loaded file into dataframe')
    del df['song']
    del df['year']
    del df['artist']
    print('Dropped columns that are not needed for analysis')
    df.dropna(subset=['genre', 'lyrics'], axis=0, inplace=True)
    print('Dropped rows with missing values')
    df = df[~df.genre.isin(["Not Available", "Other"])]
    print("Dropped rows with Not Available, Other as genre")
    # df.drop(df.loc[2:362236].index, inplace=True)  # only 2 rows for testing
    df['lyrics'] = df['lyrics'].map(lambda song: ' '.join(
        [ps.stem(tok) for tok in word_tokenize(song) if tok.isalpha() and tok.lower() in english_vocab]))
    print("Stemmed and cleaned lyrics")
    # print("Cleaned lyrics")
    df.to_csv('./clean_lyrics.csv', sep=',', encoding='utf-8', index=False)
    print('Wrote to csv file')


if __name__ == '__main__':
    main()
    exit()
