from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def tfidf_learn_vocabulary(tokenized_sentences: list[list[str]]):
    """
    Applies TF-IDF to a list of tokenized strings
    """
    vectorizer = TfidfVectorizer(analyzer='word', max_features=1000,
                                 tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    vectorizer.fit(tokenized_sentences)
    return vectorizer


def tfidf_matrix(vectorizer: TfidfVectorizer, tokenized_sentences: list[list[str]]):
    """
    Transforms a list of tokenized strings into values obtained through a previously fitted TF-IDF vectorizer
    """
    return vectorizer.transform(tokenized_sentences).toarray()


def bag_of_words_learn_vocabulary(tokenized_sentences: list[list[str]]):
    """
    Applies Bag of Words to a list of tokenized strings
    """
    vectorizer = CountVectorizer(
        max_features=1000, tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    vectorizer.fit(tokenized_sentences)
    return vectorizer


def bag_of_words_matrix(vectorizer: TfidfVectorizer, tokenized_sentences: list[list[str]]):
    """
    Transforms a list of tokenized strings into values obtained through a previously fitted Bag of Words vectorizer
    """
    return vectorizer.transform(tokenized_sentences)
