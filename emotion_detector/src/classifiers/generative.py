from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


random_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
perceptron_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                      hidden_layer_sizes=(5, 2), random_state=0)


def random_forest_fit(x_train, y_train):
    random_classifier.fit(x_train, y_train)


def random_forest_predict(x_test):
    return random_classifier.predict(x_test)


def multi_layer_perceptron_fit(x_train, y_train):
    perceptron_classifier.fit(x_train, y_train)


def multi_layer_perceptron_predict(x_test):
    return perceptron_classifier.predict(x_test)
