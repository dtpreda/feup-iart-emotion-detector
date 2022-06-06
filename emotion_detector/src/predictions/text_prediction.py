import pandas as pd
from text_analysis.preprocess import tokenize
from text_analysis.feature_extraction import tfidf_learn_vocabulary, tfidf_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def predict_text_emotion(data, algorithm):
    train_data = pd.read_csv("emotion_detector/dataset/twitter/train.csv")

    emotions = train_data.Emotion.values
    statements = train_data.drop('Emotion', axis=1).values

    statements = [' '.join(tokenize(statement[0])) for statement in statements]
    vectorizer = tfidf_learn_vocabulary(statements)
    processed_statements = tfidf_matrix(vectorizer, statements)

    predictions = algorithm(
        processed_statements, emotions, tfidf_matrix(vectorizer, [' '.join(tokenize(data))]))

    return predictions[0]
