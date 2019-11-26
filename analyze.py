import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def predict_genre(df):
    # remove genres with lowest accuracy
    x = df.loc[~df.genre.isin(['Electronic', 'R&B', 'Indie'])]['lyrics'].astype('U')
    y = df.loc[~df.genre.isin(['Electronic', 'R&B', 'Indie'])]['genre'].astype('U')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=99)

    print("linear SVC:")
    linear = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LinearSVC())
                       ])
    linear.fit(x_train, y_train)
    linear = linear.predict(x_test)
    print(accuracy_score(linear, y_test))

    print("stochastic grad descent:")
    sgd_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf',
         SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=42, max_iter=30, class_weight='balanced'))
    ])
    sgd_clf.fit(x_train, y_train)
    sgd_predicted = sgd_clf.predict(x_test)
    print(accuracy_score(y_test, sgd_predicted))
    sgd_cm = confusion_matrix(y_test, sgd_predicted)
    print(sgd_cm)  # Matrix
    sgd_cm = sgd_cm.astype('float') / sgd_cm.sum(axis=1)[:, np.newaxis]
    print(sgd_cm.diagonal())  # Accuracy scores
    print(classification_report(y_test, sgd_predicted))  # Metrics

    print("multinomial naive bayes:")
    mnb_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
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
