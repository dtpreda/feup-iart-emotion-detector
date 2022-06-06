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


def tokenize(statement: str, language: str = "english"):
    # Remove html tags
    statement = clean_html(statement)

    # Use TweetTokenizer to tokenize while preserving hashtags
    tt = nltk.tokenize.TweetTokenizer(strip_handles=True,
                                      reduce_len=True, match_phone_numbers=True)
    punctuation = list(string.punctuation)
    punctuation.remove('#')
    stop = nltk.corpus.stopwords.words(language) + punctuation
    tokens = tt.tokenize(statement)

    # Remove stop words and lowercase and lemmatize
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t.lower())
              for t in tokens if t.lower() not in stop]

    # Filter single character words
    tokens = list(filter(lambda x: len(x) > 1, tokens))

    return tokens


def bigram(tokenized_sentence: list[str]):
    return list(nltk.bigrams(tokenized_sentence))


def trigram(tokenized_sentence: list[str]):
    return list(nltk.trigrams(tokenized_sentence))


def mark_negation(tokenized_sentence: list[str]):
    return nltk.sentiment.util.mark_negation(tokenized_sentence)


def pos_tag(tokenized_sentence: list[str]):
    return nltk.pos_tag(tokenized_sentence)
