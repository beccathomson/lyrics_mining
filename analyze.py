import sys

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def predict_genre(df):
    x = df['lyrics'].astype('U')
    y = df['genre'].astype('U')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=99)

    print("sgd classifier:") # 59%
    sgd_clf = Pipeline([
        ('vect', CountVectorizer()),
        # we get worse accuracy using balanced class weight
        # ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=42, max_iter=30, class_weight='balanced'))
        ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=42, max_iter=30))
    ])
    sgd_clf.fit(x_train, y_train)
    sgd_predicted = sgd_clf.predict(x_test)
    print(accuracy_score(y_test, sgd_predicted))

    print("mnb classifier:") # 45%
    mnb_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)),
    ])
    mnb_clf.fit(x_train, y_train)
    mnb_predicted = mnb_clf.predict(x_test)
    print(accuracy_score(y_test, mnb_predicted))


def predict_year(df):
    pass


def main():
    df = pd.read_csv("./clean_lyrics.csv", header=0, index_col=0)
    predict_genre(df)
    # predict_year(df)


if __name__ == "__main__":
    main()
    sys.exit()
