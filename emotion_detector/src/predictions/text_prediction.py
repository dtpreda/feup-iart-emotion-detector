import pandas as pd
from text_analysis.preprocess import tokenize
from text_analysis.feature_extraction import tfidf_learn_vocabulary, tfidf_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def predict_text_emotion(data, algorithm, dataset_dir, rm_stop_words,
                         lowercase, lemmatize, rm_single_chars, with_bigram, with_pos_tag):
    train_data = pd.read_csv(dataset_dir + "train.csv")

    emotions = train_data.Emotion.values
    statements = train_data.drop('Emotion', axis=1).values

    statements = [tokenize(statement[0], rm_stop_words=rm_stop_words,
                           lowercase=lowercase, do_lemmatize=lemmatize, rm_single_chars=rm_single_chars, with_bigram=with_bigram, with_pos_tag=with_pos_tag)
                  for statement in statements]

    vectorizer = tfidf_learn_vocabulary(statements)
    processed_statements = tfidf_matrix(vectorizer, statements)

    predictions = algorithm(
        processed_statements, emotions, tfidf_matrix(vectorizer,
                                                     [tokenize(data, rm_stop_words=rm_stop_words,
                                                               lowercase=lowercase, do_lemmatize=lemmatize, rm_single_chars=rm_single_chars, with_bigram=with_bigram, with_pos_tag=with_pos_tag)]))

    return predictions[0]
