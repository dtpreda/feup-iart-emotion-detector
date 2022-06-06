from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


random_classifier = RandomForestClassifier(
    n_estimators=100, random_state=0)
perceptron_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                      hidden_layer_sizes=(5, 2), random_state=0)


def random_forest_fit(x_train, y_train):
    """
    Trains a Random Forest Classifier with a given data set
    """
    random_classifier.fit(x_train, y_train)


def random_forest_predict(x_test):
    """
    Predicts label data with a Random Forest Classifier for a given data set
    """
    return random_classifier.predict(x_test)


def multi_layer_perceptron_fit(x_train, y_train):
    """
    Trains a Multilayer Perceptron with a given data set
    """
    perceptron_classifier.fit(x_train, y_train)


def multi_layer_perceptron_predict(x_test):
    """
    Predicts label data with a Multilayer Perceptron for a given data set
    """
    return perceptron_classifier.predict(x_test)
