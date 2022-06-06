import string
import nltk
import nltk.sentiment.util

import re
CLEAN_HTML_REGEX = re.compile('<.*?>')


def clean_html(raw_html):
    clean_text = re.sub(CLEAN_HTML_REGEX, '', raw_html)
    return clean_text


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)


def tokenize(statement: str, language: str = "english", *, rm_stop_words=False,
             lowercase=False, lemmatize=False, rm_single_chars=False, with_bigram=False, with_pos_tag=False):
    # Use TweetTokenizer to tokenize while preserving hashtags
    statement = clean_html(statement)
    tt = nltk.tokenize.TweetTokenizer(strip_handles=True,
                                      reduce_len=True, match_phone_numbers=True)

    punctuation = list(string.punctuation)
    punctuation.remove('#')
    stop = nltk.corpus.stopwords.words(language) + punctuation
    tokens = tt.tokenize(statement)

    if rm_stop_words:
        tokens = [t for t in tokens if t not in stop]

    if lowercase:
        tokens = [t.lower() for t in tokens]

    if lemmatize:
        lemmatizer = nltk.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    if rm_single_chars:
        tokens = list(filter(lambda x: len(x) > 1, tokens))

    if with_pos_tag:
        tokens = pos_tag(tokens)

    if with_bigram:
        tokens = bigram(tokens)

    return tokens


def bigram(tokenized_sentence: list[str]):
    return list(nltk.bigrams(tokenized_sentence))


def pos_tag(tokenized_sentence: list[str]):
    return nltk.pos_tag(tokenized_sentence)
