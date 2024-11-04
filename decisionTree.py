import numpy as np
from collections import Counter

class Node: #python trick: ,*, value=None means that value has to be passed like value=x
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class Tree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        if not self.n_features:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features)
        self.root = self._build_tree(X,y)

    # _ is used before method name - means that method is private
    # recursive method to build tree
    def _build_tree(self, X, y, tree_depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #check stopping criteria
        if tree_depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        #find the best split
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        #create child nodes and call _build_tree recursively
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], tree_depth+1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], tree_depth+1)
        return Node(best_feature, best_threshold, left, right)

    @staticmethod
    def _most_common_label(y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                #calculate the information gain (entropy)
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        #parent entropy
        parent_entropy = self._entropy(y)

        #create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        #calculate the weighted average entropy of children
        n_samples = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left/n_samples) * entropy_left + (n_right/n_samples) * entropy_right

        #calculate the information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    @staticmethod
    def _split(X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    @staticmethod
    def _entropy(y):
        hist = np.bincount(y)
        ps = hist/len(y)
        sump = list()

        for p in ps:
            if p > 0:
                sump.append(p * np.log2(p))

        return -np.sum(sump)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


