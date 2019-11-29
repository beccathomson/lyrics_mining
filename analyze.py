import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def get_correlated_terms(x, y):
    n = 5
    features = x.toarray()
    genre_codes = sorted(set(y.astype(int)))
    genre_names = le.inverse_transform(genre_codes)
    for code in genre_codes:  # for each genre
        print(genre_names[code])
        features_chi2 = chi2(features, y == code)  # get features for songs in genre # works but is SUPER slow
        print("got chi features")
        indices = np.argsort(features_chi2[0])
        print("got indices")
        feature_names = np.array(tfidf.get_feature_names())[indices]
        print("Most correllated terms: " + str(genre_names[code]))
        print("\n".join(feature_names[-n:]))


def predict_genre(x, y):
    linear_clf = forest_clf = sgd_clf = bayes_clf = tree_clf = None # for testing

    x = tfidf.fit_transform(x, y) # removes stopwords based on frequency
    print("Cleaned lyrics")
    ros = RandomOverSampler(random_state=42) # necessary for multinomial bayes only
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=99)
    #
    linear_clf = LinearSVC(class_weight='balanced')
    forest_clf = RandomForestClassifier(class_weight='balanced', n_estimators=12, max_depth=40)
    sgd_clf = SGDClassifier(class_weight='balanced')
    bayes_clf = MultinomialNB()
    tree_clf = DecisionTreeClassifier(max_depth=12, class_weight='balanced')

    classifiers = {'LinearSVC': linear_clf, 'RandomForestClassifier': forest_clf, 'SGDClassifier': sgd_clf,
                   'MultinomialNB': bayes_clf,
                   'DecisionTreeClassifier': tree_clf}

    genre_codes = set(y.astype(int))
    genre_names = le.inverse_transform(list(genre_codes))

    for val in classifiers:
        if classifiers[val] is None: # for testing
            continue
        if val == 'MultinomialNB': # only oversample for multinomial
            x_train, y_train = ros.fit_resample(x_train, y_train)
        print(val)
        classifiers[val].fit(x_train, y_train)
        y_pred = classifiers[val].predict(x_test)
        print(accuracy_score(y_test, y_pred))
        print(str(genre_names))  # print labels
        genre_codes = [str(code) for code in genre_codes]
        cm = confusion_matrix(y_test, y_pred, genre_codes)
        # print(cm) # full matrix
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm.diagonal())  # Accuracy scores for each label
        # print(classification_report(y_test, classifiers[classifier]))


def main():
    global le, tfidf
    le = preprocessing.LabelEncoder()
    tfidf = TfidfVectorizer(max_df=0.8, use_idf=True)

    df = pd.read_csv("./clean_lyrics.csv", header=0)
    df = df[~df.genre.isin(['Electronic', 'R&B', 'Indie'])]  # remove genres with lowest accuracy
    df['genre'] = le.fit_transform(df.genre.values)  # encode labels
    x = df['lyrics'].astype('U')
    y = df['genre'].astype('U')

    predict_genre(x, y)
    # get_correlated_terms(x, y)


if __name__ == "__main__":
    main()
    sys.exit()
