import sys
from os import listdir
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from  sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def round_to(val, base):
    return base * round(val/base)


def main():
    df = pd.read_csv("./files/clean_lyrics.csv", header=0, index_col=0)

    i = 0
    X = []
    y = []
    for row in df.read():
            index = row[0]
            # genre = row[4]
            X.append(row[5])
            # y.append(round_to(df.loc[name, 'Score'], 5))
            y.append(df.loc[index, 'genre'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=99)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        # ('clf', MultinomialNB()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=30, tol=None))
    ])

    text_clf.fit(X_train, y_train)

    predicted = text_clf.predict(X_test)
    print(tuple(zip(predicted, y_test)))
    print(accuracy_score(y_test, predicted))


if __name__ == "__main__":
    main()
    sys.exit()
