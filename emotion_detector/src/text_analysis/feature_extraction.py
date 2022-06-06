from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def tfidf_learn_vocabulary(tokenized_sentences: list[list[str]]):
    vectorizer = TfidfVectorizer(analyzer='word', max_features=1000,
                                 tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    vectorizer.fit(tokenized_sentences)
    return vectorizer


def tfidf_matrix(vectorizer: TfidfVectorizer, tokenized_sentences: list[list[str]]):
    return vectorizer.transform(tokenized_sentences).toarray()


def bag_of_words_learn_vocabulary(tokenized_sentences: list[list[str]]):
    vectorizer = CountVectorizer(
        max_features=1000, tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    vectorizer.fit(tokenized_sentences)
    return vectorizer


def bag_of_words_matrix(vectorizer: TfidfVectorizer, tokenized_sentences: list[list[str]]):
    return vectorizer.transform(tokenized_sentences)
