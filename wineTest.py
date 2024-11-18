from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn import tree

import matplotlib.pyplot as plt
import numpy as np

from decisionTree import Tree

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred)/len(y_test)

data = datasets.load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=67
)

#implemented clasifier
my_classifier = Tree()
my_classifier.fit(X_train, y_train)
my_predictions = my_classifier.predict(X_test)

#sklearn clasifier
sklearn_classifier = tree.DecisionTreeClassifier()
sklearn_classifier.fit(X_train, y_train)
sklearn_predictions = sklearn_classifier.predict(X_test)

my_acc = accuracy(y_test, my_predictions)
print(my_acc)

print(
    f"Classification report for classifier {my_classifier}:\n"
    f"{classification_report(y_test, my_predictions)}\n"
)

labels = ["class_0","class_1","class_2"]
cm_decisionTree = confusion_matrix(y_test, my_predictions, labels=[0,1,2])
disp_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=cm_decisionTree, display_labels=labels)
disp_confusion_matrix.plot()
title = "Confusion Matrix for implemented DecisionTree algorithm, accuracy: {acc:.2f}"
title = title.format(acc = my_acc)
plt.title(title)

sklearn_acc = accuracy(y_test, sklearn_predictions)
print(sklearn_acc)

print(
    f"Classification report for classifier {sklearn_classifier}:\n"
    f"{classification_report(y_test, sklearn_predictions)}\n"
)

labels = ["class_0","class_1","class_2"]
cm_decisionTree = confusion_matrix(y_test, sklearn_predictions, labels=[0,1,2])
disp_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=cm_decisionTree, display_labels=labels)
disp_confusion_matrix.plot()
title = "Confusion Matrix for sklearn library DecisionTree algorithm, accuracy: {acc:.2f}"
title = title.format(acc = sklearn_acc)
plt.title(title)
plt.show()