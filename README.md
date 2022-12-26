# Sentiment analysis of TripAdvisor reviews

Final project for the Data Science Lab exam

In this project, a **sentiment analysis** task is carried out. A binary classification pipeline is implemented, to detect whether a certain **TripAdvisor textual review** is a "positive review" or a "negative review".

The dataset consists of 28754 reviews. The data pipeline includes the following preprocessing steps:
* Removal of non-alphanumeric characters, tokenization, case normalization, stop-words removal (data cleaning & reformatting)
* Stemming (through an Italian-based stemming algorithm)
* Bigrams extraction
* Removal of words/bigrams which are too frequent or too infrequent
* TF-IDF feature extraction
* Oversampling through SMOTE
* Feature selection through ANOVA F-test

Classification performed by a simple **Multinomial Naive Bayes** classifier on the TF-IDF text representation of each review.

**Weighted F1 score** on the test set: **0.967**

## User guide
TODO
