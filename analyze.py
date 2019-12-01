import csv
import sys

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def print_predictions(predictions, y_test):
    with open('predictions_file.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        test = list(y_test)
        test.insert(0, "Test")
        writer.writerow(test)
        print(test)
        for classifier in predictions:
            row = list(predictions[classifier])
            row.insert(0, classifier)
            print(row)
            writer.writerow(row)


def predict_genre(x, y):
    linear_clf = forest_clf = sgd_clf = bayes_clf = tree_clf = None  # for testing

    x = tfidf.fit_transform(x, y)  # removes stopwords based on frequency
    print("Cleaned lyrics")
    ros = RandomOverSampler(random_state=42)  # necessary for multinomial bayes only
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=99)

    linear_clf = LinearSVC(class_weight='balanced', dual=False)
    forest_clf = RandomForestClassifier(class_weight='balanced', n_estimators=15, max_depth=40)
    sgd_clf = SGDClassifier(alpha=0.001, class_weight='balanced', loss='log')
    bayes_clf = MultinomialNB()
    tree_clf = DecisionTreeClassifier(max_depth=12, class_weight='balanced')

    classifiers = {'LinearSVC': linear_clf, 'RandomForestClassifier': forest_clf, 'SGDClassifier': sgd_clf,
                   'MultinomialNB': bayes_clf,
                   'DecisionTreeClassifier': tree_clf}
    predictions = {}

    genre_codes = set(y.astype(int))
    genre_names = le.inverse_transform(list(genre_codes))

    for val in classifiers:
        if classifiers[val] is None:  # for testing
            continue
        if val == 'MultinomialNB':  # only oversample for multinomial
            x_train, y_train = ros.fit_resample(x_train, y_train)
        print(str(val))
        classifiers[val].fit(x_train, y_train)
        y_pred = classifiers[val].predict(x_test)
        predictions[val] = [genre_names[int(pred[0])] for pred in y_pred]

    print_predictions(predictions, [genre_names[int(true[0])] for true in y_test])


def main():
    global le, tfidf
    le = preprocessing.LabelEncoder()
    tfidf = TfidfVectorizer(max_df=0.8, use_idf=True)

    df = pd.read_csv("./clean_lyrics.csv", header=0)
    # df = df[df.genre.isin(['Hip-Hop', 'Country', 'Metal'])]  # remove genres
    df['genre'] = le.fit_transform(df.genre.values)  # encode labels
    x = df['lyrics'].astype('U')
    y = df['genre'].astype('U')

    predict_genre(x, y)


if __name__ == "__main__":
    main()
    sys.exit()
