import csv
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def main():
    try:
        os.mkdir("files")
    except Exception:
        pass

    with open("./files/raw_lyrics.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        clean_rows = [["index", "song", "year", "artist", "genre"]]
        for row in reader:
            if len(row) < 5 or len(row[5].split()) < 5:
                continue
            tokens = word_tokenize(row[5])
            clean_rows.append(row[:-1]),
            clean_rows.append(' '.join([tok.lower() for tok in tokens if tok.isalpha() and tok.lower() not in stop_words]))

    writer = csv.writer("./files/clean_lyrics.csv", delimiter=',')
    writer.write(clean_rows)


if __name__ == '__main__':
    main()
