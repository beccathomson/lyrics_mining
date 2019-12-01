import csv

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def visualize(classifier, y_pred, y_train, genre_names):
    print(classifier) # name of classifier
    print(len(y_train))
    print(len(y_pred))
    print("Accuracy: " + str(round(accuracy_score(y_train, y_pred), 5)))
    print(str(genre_names))  # print labels
    cm = confusion_matrix(y_train, y_pred, genre_names)
    print(cm)  # full matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())  # Accuracy scores for each label
    print(classification_report(y_train, y_pred))


def main():

    with open('./predictions_file.csv', 'r') as f:
        reader = csv.reader(f) # first row is test data
        y_test = next(reader)
        genre_names = set(y_test)
        for row in f:
            visualize(row[0], row[1:], y_test, genre_names)


if __name__ == "__main__":
    main()
