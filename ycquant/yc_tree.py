from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class YCRandomForest:
    def __init__(self):
        self.clf = None
        pass

    def set_params(self, n_estimators=100, max_depth=2, max_features=None, cv=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.cv = cv

    def save(self):
        pass

    def load(self):
        pass

    def predict(self, X):
        return self.clf.predict(X)

    def predict_prob(self, X):
        return self.clf.predict_prob(X)

    def fit(self, X, y):
        if not hasattr(self, 'n_estimators'):
            print("Automatically initilizing....")
            self.set_params()
        self.clf = GridSearchCV(estimator=RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, max_features=None),
                                cv=self.cv)

        self.clf.fit(X, y)

        return self.clf
