import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
df=pd.read_csv(r"E:\Downloads\SocialNetworkAds.csv")
x=df.iloc[:,[2,3]].values
y=df.iloc[:, 4].values
# sns.heatmap(df.corr())
# plt.show()
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)
transformer=StandardScaler()## standardize the data ranging near to zero by substracting by mean and divide by standard deviation. Normalization is gettting values either to 0 or 1.
X_train=transformer.fit_transform(X_train)
X_test=transformer.fit_transform(X_test)
lr=LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
X_set,y_set=X_test,y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min() -1, stop=X_set[:,0].max() +1,step=0.01),
                 np.arange(start=X_set[:,1].min() -1, stop=X_set[:,1].max() +1,step=0.01))
plt.contourf(X1,X2, lr.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(("red","green"))(i),label=j)
plt.title("Logistic Regression(Training set)")
plt.xlabel("Age")
# plt.ylabel("Estimated Salary")
# plt.legend()
# plt.show()    
cm=confusion_matrix(y_test,y_pred)## matrix to determine the TP,FP,TN,FN to determine the accuracy by adding the diagonal value by total number of values
print("mae:",metrics.mean_absolute_error(y_test,y_pred))
print("mse:",metrics.mean_squared_error(y_test,y_pred))
print("RMSE:", np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))