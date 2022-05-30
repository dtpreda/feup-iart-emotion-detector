import pandas as pd
import nltk
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pd.read_csv("emotion_detector/asset/dataset/emotions.csv")
test_data = pd.read_csv(
    "emotion_detector/asset/dataset/testing_data.csv", encoding="cp1252")
nltk.download('stopwords')


def preprocessing(statement):
    tt = TweetTokenizer(strip_handles=True, match_phone_numbers=False)
    new_statement = tt.tokenize(statement)
    new_statement = [x.lower() for x in new_statement if len(
        x) > 1 and not x in stopwords.words("english")]
    return new_statement


emotions = train_data.Emotion.values
statements = train_data.drop('Emotion', axis=1).values

statements = [preprocessing(statement[0]) for statement in statements]

vectorizer = TfidfVectorizer(max_features=2500, stop_words=stopwords.words(
    "english"), analyzer='word', tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)

processed_statements = vectorizer.fit_transform(statements)

x_train, x_test, y_train, y_test = train_test_split(
    processed_statements, emotions, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=200, random_state=0)
classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)

print(
    f"Accuracy on 20% of the train dataset: {accuracy_score(y_test, predictions)}")

test_emotions = test_data.Emotion.values
test_statements = test_data.drop('Emotion', axis=1).values

test_statements = [preprocessing(statement[0])
                   for statement in test_statements]
processed_test_statements = vectorizer.fit_transform(test_statements)

# x_train, x_test, y_train, y_test = train_test_split(processed_test_statements, test_emotions, test_size=1.0, random_state=0)

# print(test_statements)

predictions = classifier.predict(processed_test_statements)

print(
    f"Accuracy on the test dataset: {accuracy_score(test_emotions, predictions)}")
