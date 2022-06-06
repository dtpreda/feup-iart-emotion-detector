from sklearn.naive_bayes import GaussianNB, MultinomialNB

gaussian_classifier = GaussianNB()
multinomial_classifier = MultinomialNB()


def gaussian_naive_bayes_fit(x_train, y_train):
    gaussian_classifier.fit(x_train, y_train)


def gaussian_naive_bayes_predict(x_test):
    return gaussian_classifier.predict(x_test)


def multinomial_naive_bayes_fit(x_train, y_train):
    multinomial_classifier.fit(x_train, y_train)


def multinomial_naive_bayes_predict(x_test):
    return multinomial_classifier.predict(x_test)
