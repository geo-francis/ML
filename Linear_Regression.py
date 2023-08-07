import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
df=pd.read_csv(r"E:\Downloads\Salary_Data.csv")
x=df.iloc[:, :-1].values
y=df.iloc[:, 1].values
# sns.heatmap(df.corr())
# plt.show()
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
print(y_pred)
plt.scatter(X_test,y_test,color="blue")
plt.plot(X_test,lr.predict(X_test),color="red")
plt.title("salary expectations with experience")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
# plt.show()
print("mae:",metrics.mean_absolute_error(y_test,y_pred))
print("mse:",metrics.mean_squared_error(y_test,y_pred))
print("RMSE:", np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))
print("working")