from time import time
import pandas as pd
from text_analysis.preprocess import tokenize
from text_analysis.feature_extraction import tfidf_learn_vocabulary, tfidf_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def predict_dataset(fit_algorithm, predict_algorithm, dataset_dir, rm_stop_words,
                    lowercase, lemmatize, rm_single_chars, with_bigram, with_pos_tag):

    start = time()

    train_data = pd.read_csv(dataset_dir + "train.csv")
    test_data = pd.read_csv(dataset_dir + "test.csv", encoding="cp1252")

    emotions = train_data.Emotion.values
    statements = train_data.drop('Emotion', axis=1).values

    statements = [tokenize(statement[0], rm_stop_words=rm_stop_words,
                           lowercase=lowercase, do_lemmatize=lemmatize, rm_single_chars=rm_single_chars, with_bigram=with_bigram, with_pos_tag=with_pos_tag) for statement in statements]
    vectorizer = tfidf_learn_vocabulary(statements)
    processed_statements = tfidf_matrix(vectorizer, statements)

    x_train, x_test, y_train, y_test = train_test_split(
        processed_statements, emotions, test_size=0.2, random_state=0)

    fit_algorithm(x_train, y_train)

    train_predictions = predict_algorithm(x_test)

    test_emotions = test_data.Emotion.values
    test_statements = test_data.drop('Emotion', axis=1).values

    test_statements = [tokenize(statement[0], rm_stop_words=rm_stop_words,
                                lowercase=lowercase, do_lemmatize=lemmatize, rm_single_chars=rm_single_chars, with_bigram=with_bigram, with_pos_tag=with_pos_tag)
                       for statement in test_statements]
    processed_test_statements = tfidf_matrix(vectorizer, test_statements)

    test_predictions = predict_algorithm(processed_test_statements)

    return (round(accuracy_score(y_test, train_predictions), 3), round(accuracy_score(test_emotions, test_predictions), 3), str(round(time() - start, 3)))
