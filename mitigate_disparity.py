import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = [[75, 25], [12, 88]]
df_cm = pd.DataFrame(array, index = ["True", "False"],
              columns = ["Normal", "Demented"])
plt.figure(figsize = (10,7))
plt.title("Testing")
sn.heatmap(df_cm, annot=True,cmap="OrRd")
