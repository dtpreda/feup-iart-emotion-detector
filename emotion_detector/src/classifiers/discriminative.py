from sklearn.naive_bayes import GaussianNB, MultinomialNB

gaussian_classifier = GaussianNB()
multinomial_classifier = MultinomialNB()


def gaussian_naive_bayes_fit(x_train, y_train):
    """
    Trains a Gaussian Naive Bayes with a given data set
    """
    gaussian_classifier.fit(x_train, y_train)


def gaussian_naive_bayes_predict(x_test):
    """
    Predicts label data with a Gaussian Naive Bayes for a given data set
    """
    return gaussian_classifier.predict(x_test)


def multinomial_naive_bayes_fit(x_train, y_train):
    """
    Trains a Multinomial Naive Bayes with a given data set
    """
    multinomial_classifier.fit(x_train, y_train)


def multinomial_naive_bayes_predict(x_test):
    """
    Predicts label data with a Multinomial Naive Bayes for a given data set
    """
    return multinomial_classifier.predict(x_test)
