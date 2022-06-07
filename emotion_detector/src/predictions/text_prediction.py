import pandas as pd
from text_analysis.preprocess import tokenize
from text_analysis.feature_extraction import tfidf_learn_vocabulary, tfidf_matrix

previous_dataset_dir = None
previous_predict_algorithm = None
previous_rm_stop_words = None
previous_lowercase = None
previous_lemmatize = None
previous_rm_single_chars = None
previous_with_bigram = None
previous_with_pos_tag = None


def predict_text_emotion(data, fit_algorithm, predict_algorithm, dataset_dir, rm_stop_words,
                         lowercase, lemmatize, rm_single_chars, with_bigram, with_pos_tag):
    """
    Trains a model with a dataset and then predicts the underlying emotion of the given statement.
    If the desired model was already trained, just predicts the emotion for the statemnt.
    """
    train_data = pd.read_csv(dataset_dir + "train.csv")

    emotions = train_data.Emotion.values
    statements = train_data.drop('Emotion', axis=1).values

    statements = [tokenize(statement[0], rm_stop_words=rm_stop_words,
                           lowercase=lowercase, do_lemmatize=lemmatize, rm_single_chars=rm_single_chars, with_bigram=with_bigram, with_pos_tag=with_pos_tag)
                  for statement in statements]

    vectorizer = tfidf_learn_vocabulary(statements)
    processed_statements = tfidf_matrix(vectorizer, statements)

    global previous_dataset_dir
    global previous_predict_algorithm
    global previous_rm_stop_words
    global previous_lowercase
    global previous_lemmatize
    global previous_rm_single_chars
    global previous_with_bigram
    global previous_with_pos_tag

    if previous_dataset_dir != dataset_dir or previous_predict_algorithm != predict_algorithm \
            or previous_rm_stop_words != rm_stop_words or previous_lowercase != lowercase \
            or previous_lemmatize != lemmatize or previous_with_bigram != with_bigram \
            or previous_rm_single_chars != rm_single_chars or previous_with_pos_tag != with_pos_tag:
        fit_algorithm(processed_statements, emotions)
        previous_dataset_dir = dataset_dir
        previous_predict_algorithm = predict_algorithm
        previous_rm_stop_words = rm_stop_words
        previous_lowercase = lowercase
        previous_lemmatize = lemmatize
        previous_rm_single_chars = rm_single_chars
        previous_with_bigram = with_bigram
        previous_with_pos_tag = with_pos_tag

    predictions = predict_algorithm(
        tfidf_matrix(vectorizer, [tokenize(data, rm_stop_words=rm_stop_words,
                                           lowercase=lowercase, do_lemmatize=lemmatize, rm_single_chars=rm_single_chars, with_bigram=with_bigram, with_pos_tag=with_pos_tag)]))

    return predictions[0]
