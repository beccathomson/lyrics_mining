import csv

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def visualize(classifier, y_pred, y_test, genre_names):
    print("\n"+str(classifier)) # name of classifier
    print("Accuracy: " + str(round(accuracy_score(y_test, y_pred), 5)))
    print(str(genre_names))  # print labels
    cm = confusion_matrix(y_test, y_pred, genre_names)
    print(cm)  # full matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())  # Accuracy scores for each label
    print(classification_report(y_test, y_pred))


def main():

    with open('./predictions_file.csv', 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        y_test = next(reader) # first row is test data
        genre_names = sorted(set(y_test[1:]))
        for row in reader:
            if row[0] == "Test":
                continue
            visualize(row[0], row[1:], y_test[1:], genre_names)


if __name__ == "__main__":
    main()
