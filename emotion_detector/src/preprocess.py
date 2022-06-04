import string
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)


def tokenize(statement: str, language: str = "english"):
    # Use TweetTokenizer to tokenize while preserving hashtags
    tt = nltk.tokenize.TweetTokenizer(strip_handles=True,
                                      reduce_len=True, match_phone_numbers=True)
    stop = nltk.corpus.stopwords.words(language) + list(string.punctuation)
    tokens = tt.tokenize(statement)

    # Remove stop words and lowercase and lemmatize
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t.lower())
              for t in tokens if t.lower() not in stop]

    # Filter single character words
    tokens = list(filter(lambda x: len(x) > 1, tokens))

    return tokens
