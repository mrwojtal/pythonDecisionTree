from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import matplotlib.pyplot as plt
import numpy as np

from decisionTree import Tree

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, random_state=45
)

classifier = Tree()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred)/len(y_test)

acc = accuracy(y_test, predictions)
print(acc)

print(
    f"Classification report for classifier {classifier}:\n"
    f"{classification_report(y_test, predictions)}\n"
)

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cm_decisionTree = confusion_matrix(y_test, predictions, labels=labels)
disp_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=cm_decisionTree, display_labels=labels)
disp_confusion_matrix.plot()
plt.show()