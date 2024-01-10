#Divya Darshi
#1002090905

import numpy as np

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)

    def predict(self, X):
        predictions = [self._predict(x, self.tree) for x in X]
        return np.array(predictions)

    def build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if depth == self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = self.most_common_class(y)
            return leaf_value

        best_criteria, best_split = self.find_best_split(X, y)

        if best_split is None:  # Handle the case when no split is found
            leaf_value = self.most_common_class(y)
            return leaf_value

        left_indices = best_split['left'][0]
        right_indices = best_split['right'][0]

        left_subtree = self.build_tree(X[left_indices], y[left_indices], depth=depth + 1)
        right_subtree = self.build_tree(X[right_indices], y[right_indices], depth=depth + 1)

        return (best_criteria, left_subtree, right_subtree)

    def find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_criteria = None
        best_gini = float('inf')
        best_split = None

        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            for value in unique_values:
                criteria = (feature_idx, value)
                left, right = self.split(X, y, feature_idx, value)
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue

                gini = self.calculate_gini(left, right, y)
                if gini < best_gini:
                    best_criteria = criteria
                    best_gini = gini
                    best_split = {'left': (left, y[left]), 'right': (right, y[right])}

        return best_criteria, best_split

    def split(self, X, y, feature_idx, threshold):
        left = np.where(X[:, feature_idx] <= threshold)[0]
        right = np.where(X[:, feature_idx] > threshold)[0]
        return left, right

    def calculate_gini(self, left, right, y):
        n_left = len(left)
        n_right = len(right)
        n_total = n_left + n_right

        gini_left = 1.0 - sum((np.sum(y[left] == c) / n_left) ** 2 for c in np.unique(y))
        gini_right = 1.0 - sum((np.sum(y[right] == c) / n_right) ** 2 for c in np.unique(y))

        gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        return gini

    def most_common_class(self, y):
        return np.bincount(y).argmax()

    def _predict(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature_idx, threshold = tree[0]
        if x[feature_idx] <= threshold:
            return self._predict(x, tree[1])
        else:
            return self._predict(x, tree[2])

class RandomForest:
    def __init__(self, num_learners=50, min_features=None, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.num_learners = num_learners
        self.min_features = min_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.num_learners):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sampled = X[sample_indices]
            y_sampled = y[sample_indices]

            selected_features = self.features_selection(n_features)

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_sampled[:, selected_features], y_sampled)
            self.trees.append((tree, selected_features))

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)), dtype=int)
        for i, (tree, selected_features) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, selected_features])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)

    def features_selection(self, n_features):
        if self.min_features is None:
            return np.arange(n_features)
        else:
            return np.random.choice(n_features, size=self.min_features, replace=False)

class AdaBoost:
    def __init__(self, weak_learner=None, num_learners=50, learning_rate=0.1, random_state=None):
        self.weak_learner = weak_learner
        self.num_learners =num_learners
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.learners = []
        self.learner_weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, 1 / n_samples)

        for _ in range(self.num_learners):
            if isinstance(self.weak_learner, type):
                learner = self.weak_learner()  # Create an instance of the base estimator
            else:
                learner = self.weak_learner  # Use the provided instance of the base estimator

            learner.fit(X, y)
            y_pred = learner.predict(X)

            error = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights)

            if error >= 0.5:
                break

            learner_weight = self.learning_rate * np.log((1 - error) / error)

            sample_weights *= np.exp(-learner_weight * y * y_pred)
            sample_weights /= np.sum(sample_weights)

            self.learners.append(learner)
            self.learner_weights.append(learner_weight)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for learner, learner_weight in zip(self.learners, self.learner_weights):
            learner_predictions = learner.predict(X)
            predictions += learner_weight * learner_predictions

            # Check for early stopping based on perfect fit
            if np.all(
                    learner_predictions == predictions):  # If all learner predictions are the same as the current ensemble predictions
                break

        return np.sign(predictions)
