from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import matplotlib.pyplot as plt
import numpy as np

from decisionTree import Tree

data = datasets.load_wine()
X, y = data.data, data.target
#print(X)
# print(y)
# print(data.target_names)
# print(np.shape(X))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=67
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

labels = ["class_0","class_1","class_2"]
cm_decisionTree = confusion_matrix(y_test, predictions, labels=[0,1,2])
disp_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=cm_decisionTree, display_labels=labels)
disp_confusion_matrix.plot()
plt.show()