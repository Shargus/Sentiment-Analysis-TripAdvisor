##
from time import time

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.stem import SnowballStemmer

import csv
from collections import Counter
import string

##
# DATA EXPLORATION

print('\n\n\n=============== DATA EXPLORATION ===============')

data_dev = pd.read_csv("dataset/development.csv")
reviews = data_dev['text']
labels = data_dev['class']

print("DEVELOPMENT SET")
print("Number of reviews in the development set:", len(reviews))
print("Number of 'pos' and 'neg':", [(k, v) for k, v in Counter(labels).items()])
print("Number of empty reviews:", sum([1 for doc in reviews if doc == ""]))
print("Min and max length of reviews:", min([len(doc) for doc in reviews]), max([len(doc) for doc in reviews]), '\n')

data_eval = pd.read_csv("dataset/evaluation.csv")
reviews_eval = data_eval['text']

print("EVALUATION SET")
print("Number of reviews:", len(reviews))
print("Number of 'pos' and 'neg':", [(k, v) for k, v in Counter(labels).items()])
print("Number of empty reviews:", sum([1 for doc in reviews_eval if doc == ""]))
print("Min and max length of reviews:", min([len(doc) for doc in reviews_eval]), max([len(doc) for doc in reviews_eval]), '\n')


# split in training + test set
print('Splitting the development set in TRAINING and TEST set (95%/5%)\n')
reviews_train, reviews_test, labels_train, labels_test = train_test_split(
    reviews, labels, test_size=0.05, stratify=labels)   #random_state=3

print("Number of reviews in the training set:", len(reviews_train))
print("Number of 'pos' and 'neg' in training set:", [(k, v) for k, v in Counter(labels_train).items()])
print("Number of reviews in the test set:", len(reviews_test))
print("Number of 'pos' and 'neg' in test set:", [(k, v) for k, v in Counter(labels_test).items()], '\n')

c = Counter(labels_train)
for i in ["pos", "neg"]:
    print(f"Training set: fraction of {i} = {c[i]/len(labels_train):.5f}")

c = Counter(labels_test)
for i in ["pos", "neg"]:
    print(f"Test set: fraction of {i} = {c[i]/len(labels_test):.5f}")


##
# PREPROCESSING

print('\n\n\n=============== PREPROCESSING ===============')

def tokenize_and_stem(doc):
    """
    Keeps only letters from a to z and the letters 'à','è','é','ì','ò','ó','ù' and removes stopwords,
    converts to lowercase and finally stem the resulting tokens.

    Input: string (text)
    Output: string with the tokenized and stemmed text
    """
    doc = doc.replace("\n", " ").replace("\t", " ")
    to_keep = {'à', 'è', 'é', 'ì', 'ò', 'ó', 'ù'}.union(set(string.ascii_lowercase))
    new_doc = ''.join([char.lower() if (char.lower() in to_keep) else ' ' for char in doc ])
    return ' '.join([stemmer.stem(token.lower()) for token in new_doc.split(" ") if (token and token not in stopwords)])


stemmer = SnowballStemmer("italian")
stopwords = set(get_stop_words('it'))

print('Preprocessing of the training and test sets and grid search has begun')

t0 = time()

# Tokenization, case normalization, stemming
reviews_train = [tokenize_and_stem(doc) for doc in reviews_train]
reviews_test = [tokenize_and_stem(doc) for doc in reviews_test]

# Parameters for the grid search. Other values (not shown here) have also been tested.
params_grid = {
    'vectorizer__max_df' : [0.03, 0.06, 0.1, 0.2],
    'vectorizer__min_df' : [2, 3, 4, 5],
    'selector__percentile' : [15, 18, 21, 25]
}

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1,2))),
    ('oversampler', SMOTE()),   #random_state=3
    ('selector', SelectPercentile(f_classif)),
    ('clf', MultinomialNB())
])

# Grid search with 5-fold cross-validation
gridsearch = GridSearchCV(pipeline, params_grid, scoring='f1_weighted', cv=5)
_ = gridsearch.fit(reviews_train, labels_train)

print(f'Preprocessing on training and test sets and grid search done.\nElapsed time: {t0:.2f}')

print("Best parameter values found:", gridsearch.best_params_)
print("Weighted F1 score of the best estimator: %.5f" % gridsearch.best_score_)


##
# TRAINING OF THE BEST ESTIMATOR AND TEST ON THE TEST SET

print('\n\n\n=============== TRAINING AND TEST ===============')

final_model = gridsearch.best_estimator_

# TRAINING: fit on the WHOLE training set
final_model.fit(reviews_train, labels_train)

# TEST: predict on the test set and on the training set
labels_train_pred = final_model.predict(reviews_train)
labels_test_pred = final_model.predict(reviews_test)


# PERFORMANCE REPORT (weighted f1 on training and test sets)
f1_train = f1_score(labels_train, labels_train_pred, average='weighted')
print("Weighted f1 score on the training set: %.5f" % f1_train)
f1_test = f1_score(labels_test, labels_test_pred, average='weighted')
print("Weighted f1 score on the test set: %.5f" % f1_test)


##
# EVALUATION

print('\n\n\n=============== PREDICTION ON EVALUATION DATASET ===============')

reviews_eval = [tokenize_and_stem(doc) for doc in reviews_eval]

labels_eval = final_model.predict(reviews_eval)

print('Done!')


##
# Some DATA VISUALIZATION
conf_mat = confusion_matrix(labels_test, labels_test_pred)
conf_mat_df = pd.DataFrame(conf_mat, index = ['neg','pos'], columns = ['neg','pos'])
conf_mat_df.index.name = 'Actual'
conf_mat_df.columns.name = 'Predicted'
sns.heatmap(conf_mat_df, annot=True, cmap='GnBu', annot_kws={"size": 16}, fmt='g', cbar=False)
plt.show()


##
# PRINTING

print('\n\n\n=============== PRINTING OF "submission.csv" ===============')

with open('dataset/PREDICTED_LABELS.csv', 'w') as myfile:
    wr = csv.writer(myfile, lineterminator='\n')
    wr.writerow(["Id", "Predicted"])
    i=0     #counter
    for l in labels_eval:
        wr.writerow([i,l])
        i+=1

print('Done!')