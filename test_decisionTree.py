import unittest
import numpy as np
from decisionTree import Node, Tree

class TestNode(unittest.TestCase):
    def test_is_leaf_node(self):
        # Test dla liścia
        node_leaf = Node(value=10)
        self.assertTrue(node_leaf.is_leaf_node())

        # Test dla węzła niebędącego liściem
        node_non_leaf = Node(feature=0, threshold=5)
        self.assertFalse(node_non_leaf.is_leaf_node())


class TestTree(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])
        self.tree = Tree(max_depth=3, min_samples_split=2)

    def test_fit(self):
        self.tree.fit(self.X, self.y)
        self.assertIsNotNone(self.tree.root)
        self.assertTrue(isinstance(self.tree.root, Node))

    def test_predict(self):
        self.tree.fit(self.X, self.y)
        predictions = self.tree.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

    def test_entropy(self):
        y = np.array([1, 1, 0, 0])
        entropy = self.tree._entropy(y)
        self.assertAlmostEqual(entropy, 1.0)

        y = np.array([0, 0, 0, 0])
        entropy = self.tree._entropy(y)
        self.assertAlmostEqual(entropy, 0.0)

    def test_split(self):
        X_column = np.array([1, 2, 3, 4, 5])
        split_threshold = 3
        left_idxs, right_idxs = self.tree._split(X_column, split_threshold)

        self.assertTrue(np.array_equal(left_idxs, [0, 1, 2]))
        self.assertTrue(np.array_equal(right_idxs, [3, 4]))

    def test_information_gain(self):
        y = np.array([0, 0, 1, 1])
        X_column = np.array([1, 2, 3, 4])
        threshold = 2.5

        gain = self.tree._information_gain(y, X_column, threshold)
        self.assertGreaterEqual(gain, 0)

    def test_most_common_label(self):
        y = np.array([0, 0, 1, 1, 0])
        most_common = self.tree._most_common_label(y)
        self.assertEqual(most_common, 0)

    def test_traverse_tree(self):
        node_left = Node(value=0)
        node_right = Node(value=1)
        root = Node(feature=0, threshold=2, left=node_left, right=node_right)

        self.tree.root = root
        pred = self.tree._traverse_tree(np.array([1]), self.tree.root)
        self.assertEqual(pred, 0)

        pred = self.tree._traverse_tree(np.array([3]), self.tree.root)
        self.assertEqual(pred, 1)


if __name__ == "__main__":
    unittest.main()
