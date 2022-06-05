from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def tfidf_matrix(tokenized_sentences: list[list[str]]):
    vectorizer = TfidfVectorizer(max_features=1000, analyzer='word',
                                 tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    return vectorizer.fit_transform(tokenized_sentences).toarray()


def bag_of_words_matrix(tokenized_sentences: list[list[str]]):
    vectorizer = CountVectorizer(
        max_features=1000, tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    return vectorizer.fit_transform(tokenized_sentences)
