import matplotlib.pyplot as plt
import numpy
import sklearn
from sklearn import metrics

list1 = [1] * 42
list1 += [0] * 8
list1 += [1] * 10
list1 += [0] * 40
list2 = [1] * 50
list2 += [0] * 50



confusion_matrix = metrics.confusion_matrix(list1, list2)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["True", "False"])

cm_display.plot()
plt.show()
