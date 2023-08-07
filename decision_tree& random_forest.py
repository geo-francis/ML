import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
df=pd.read_csv(r"E:\Downloads\kyphosis.csv")
x=df.iloc[:, 1:4].values
y=df.iloc[:, 0].values
sns.barplot(x="Kyphosis",y="Age",data=df)
sns.pairplot(df,hue="Kyphosis",palette="Set1")
# plt.show()
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=100)## random state shuffling of sets.
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_test)
y_pred
cm=confusion_matrix(y_test,y_pred)
cm
## using random forest - where multiple decision trees
rtree=RandomForestClassifier(n_estimators=100)## no oof estimators is no of decsion trees
rtree.fit(X_train,y_train)
rtree_pred=rtree.predict(X_test)
rtree_pred
cm=confusion_matrix(y_test,rtree_pred)
cm