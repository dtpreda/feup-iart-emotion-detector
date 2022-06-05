from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def random_forest_predict(x_train, y_train, x_test):
    classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    classifier.fit(x_train, y_train)
    return classifier.predict(x_test)


def multi_layer_perceptron_predict(x_train, y_train, x_test):
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(5, 2), random_state=1)
    classifier.fit(x_train, y_train)
    return classifier.predict(x_test)
