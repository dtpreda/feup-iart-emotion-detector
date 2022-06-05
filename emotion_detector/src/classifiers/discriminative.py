from sklearn.naive_bayes import GaussianNB, MultinomialNB


def gaussian_naive_bayes_predict(x_train, y_train, x_test):
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    return classifier.predict(x_test)


def multinomial_naive_bayes_predict(x_train, y_train, x_test):
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)
    return classifier.predict(x_test)
