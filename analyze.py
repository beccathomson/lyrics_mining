import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_log_error, \
    mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def do_linear(x_train, x_test, y_train, y_test):
    print("linear SVC:")
    lin_predicted = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', LinearSVC(class_weight='balanced'))
                              ])
    lin_predicted.fit(x_train, y_train)
    lin_predicted = lin_predicted.predict(x_test)
    print(accuracy_score(lin_predicted, y_test))
    return lin_predicted


def do_grad_descent(x_train, x_test, y_train, y_test):
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
    return sgd_predicted


def do_bayes(x_train, x_test, y_train, y_test):
    print("multinomial naive bayes:")
    mnb_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)),
    ])
    mnb_clf.fit(x_train, y_train)
    mnb_predicted = mnb_clf.predict(x_test)
    print(accuracy_score(y_test, mnb_predicted))
    return mnb_predicted


def do_metrics(y_test, classifiers):
    for classifier in classifiers:
        print(classifier)
        sgd_cm = confusion_matrix(y_test, classifiers[classifier])
        # print(sgd_cm) # full matrix
        sgd_cm = sgd_cm.astype('float') / sgd_cm.sum(axis=1)[:, np.newaxis]
        print(sgd_cm.diagonal())  # Accuracy scores for each label
        # print(classification_report(y_test, classifiers[classifier]))  # Metrics


def predict_genre(df):
    # remove genres with lowest accuracy
    x = df.loc[~df.genre.isin(['Electronic', 'R&B', 'Indie'])]['lyrics'].astype('U')
    y = df.loc[~df.genre.isin(['Electronic', 'R&B', 'Indie'])]['genre'].astype('U')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=99)

    forest_predicted = do_random_forest(x_train, x_test, y_train, y_test)
    lin_predicted = do_linear(x_train, x_test, y_train, y_test)
    sgd_predicted = do_grad_descent(x_train, x_test, y_train, y_test)
    bayes_predicted = do_bayes(x_train, x_test, y_train, y_test)  # TODO: oversampling for bayes (no class_weight param)

    do_metrics(y_test, {'linear SVC': lin_predicted, 'grad descent': sgd_predicted, 'bayes': bayes_predicted,
                        'forest': forest_predicted})


def do_random_forest(x_train, x_test, y_train, y_test):
    print("random forest classifier:")
    forest_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf',
         RandomForestClassifier(n_estimators=10, class_weight='balanced', max_depth=50))
    ])
    forest_clf.fit(x_train, y_train)
    forest_predicted = forest_clf.predict(x_test)
    print(mean_squared_error(y_test, forest_predicted))
    print(mean_squared_log_error(y_test, forest_predicted))
    print(accuracy_score(y_test, forest_predicted))
    return forest_predicted


def main():
    df = pd.read_csv("./clean_lyrics.csv", header=0)
    # encode labels
    le = preprocessing.LabelEncoder()
    df['genre'] = le.fit_transform(df.genre.values)
    predict_genre(df)


if __name__ == "__main__":
    main()
    sys.exit()
