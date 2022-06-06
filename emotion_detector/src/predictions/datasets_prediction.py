import pandas as pd
from text_analysis.preprocess import tokenize
from text_analysis.feature_extraction import tfidf_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from classifiers.generative import random_forest_predict, multi_layer_perceptron_predict
from classifiers.discriminative import gaussian_naive_bayes_predict, multinomial_naive_bayes_predict


def predict_dataset(algorithm, dataset_dir):

    train_data = pd.read_csv(dataset_dir + "train.csv")
    test_data = pd.read_csv(dataset_dir + "test.csv", encoding="cp1252")

    emotions = train_data.Emotion.values
    statements = train_data.drop('Emotion', axis=1).values

    statements = [tokenize(statement[0]) for statement in statements]
    processed_statements = tfidf_matrix(statements)

    x_train, x_test, y_train, y_test = train_test_split(
        processed_statements, emotions, test_size=0.2, random_state=0)

    train_predictions = algorithm(
        x_train, y_train, x_test)

    # print(
    #    f"Accuracy on 20% of the train dataset: {accuracy_score(y_test, train_predictions)}")

    test_emotions = test_data.Emotion.values
    test_statements = test_data.drop('Emotion', axis=1).values

    test_statements = [tokenize(statement[0])
                       for statement in test_statements]
    processed_test_statements = tfidf_matrix(test_statements)

    test_predictions = algorithm(
        x_train, y_train, processed_test_statements)

    # print(
    #    f"Accuracy on the test dataset: {accuracy_score(test_emotions, test_predictions)}")

    return (accuracy_score(y_test, train_predictions), accuracy_score(test_emotions, test_predictions))
