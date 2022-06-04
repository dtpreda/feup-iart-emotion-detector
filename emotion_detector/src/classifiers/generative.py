from sklearn.ensemble import RandomForestClassifier


def random_forest_predict(x_train, y_train, x_test):
    classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    classifier.fit(x_train, y_train)
    return classifier.predict(x_test)
