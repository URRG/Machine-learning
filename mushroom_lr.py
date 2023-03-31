import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns

# imported all the important libraires 
df = pd.read_csv("/home/utkarsh/Documents/VS code/Python/mushrooms.csv")
df.info()
# just read the csv file
from sklearn.preprocessing import LabelEncoder
mush = df.copy()
le = LabelEncoder()
for col in mush.columns:
  mush[col] = le.fit_transform(mush[col])
# used label encoder so that the datset could be converted into binary

poi = 0                                    # applied a loop to count the occourance of each
edi = 0
for index in range(8124):
    if df.iloc[index, 0] == 'e':
        edi += 1
    else:
        poi += 1

fig = plot.figure(figsize = (10, 10))
plot.bar(['mushroom class P', 'mushroom class E'], [poi, edi], color=['#000023', '#FFF243'])
plot.xlabel('Class mushroom belongs to')
plot.ylabel('Datapoints')
plot.title('Check for balanced or imbalanced dataset')
plot.show()
#plotted a bar graph to figure out whether the dataset is balanced or imbalanced and it turned out that the dataset is not balanced
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
y = mush["class"].values
x = mush.drop(["class"],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size = 0.25)
# used train test split to break the data so that it becomes suitable for application of confusion matrix

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression as lr
y_pred_lr = lr.predict(x_test)
y_true_lr = y_test
cm = confusion_matrix(y_true_lr, y_pred_lr)
f, ax = plot.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plot.xlabel("y_pred_lr")
plot.ylabel("y_true_lr")
plot.show()
