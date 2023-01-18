import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = [[75, 25], [12, 88]]
df_cm = pd.DataFrame(array, index = ["True", "False"],
              columns = ["Normal", "Demented"])
plt.figure(figsize = (10,7))
plt.title("Testing")
sn.heatmap(df_cm, annot=True,cmap="OrRd")

# 1. Inputs

# (1) A model development dataset that contains information on:

# (i)    Model features

# (ii)   Model label

# (iii)  Sample weights

# (iv)  Demographic data on protected and reference classes

# 2. Outputs

# (1)  The fair/debiased model object, taking the form of a sklearn-style python object with the following functions accessible:

# (i)    .fit() – trains the model

# (ii)   .predict() / .predict_proba() – makes predictions using new data

# (iii)  .transform() – filters or modifies input data, if applicable

# (2)  [Optional] graphics/visualization, useful formatted output
