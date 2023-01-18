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

# 1. Inputs

# (1)  A dataframe with one row per individual. Columns will include:

# (i)    Model prediction (as a probability)

# (ii)   Binary outcome (i.e. 0 or 1, where 1 indicates the favorable outcome for the individual being scored)

# (iii)   Model label

# (iv)  Sample weights

# (v)  Demographic data on protected and reference classes

# 2. Outputs

# (1)  One value per protected class measuring discrimination for each metric used

# (2)  [Optional] graphics/visualization, useful formatted output
