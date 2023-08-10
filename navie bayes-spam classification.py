import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import MultinomialNB #used to model the probabilities of multiple possible outcomes.
df=pd.read_csv(r"C:\Users\user\Project 6\emails.csv")
df.shape
df_new=df[["text","spam"]]
df_new['spam'] = pd.to_numeric(df_new['spam'], errors='coerce')  # Convert to numeric, replacing non-numeric with NaN
df_new = df_new.dropna(subset=['spam'])
df_new['spam'] = df_new['spam'].astype(int)
df_new.info()

## visualize the dataset

normal_mail=df_new[df_new['spam']==0]
spam=df_new[df_new['spam']==1]
# spam_percentage=((len(spam)/len(df_new))*100)
# spam_percentage
# normal_mail_percent=((len(normal_mail)/len(df_new))*100)
# normal_mail_percent 
  # Set the figure size
sns.countplot(data=df_new, x='spam')
plt.show()

## Training and Testing data

## use of count vectorizer, is a technique used to convert text data into numerical vectors
## tokenizer

# sample=["everything is good","this is good","that is bad and that is good"]
# count=CountVectorizer()
# sample_result=count.fit_transform(sample)
# sample_result.toarray()#array([[0, 1, 1, 1, 0, 0],--"everything is good"--position of verything is 2 so 2 positions is 1
#     #                          [0, 0, 1, 1, 0, 1],
#     #                          [1, 0, 0, 2, 2, 0]]-- count of that and is 2 so that value is 2 in the respective position

# count.get_feature_names_out()#array(['and','bad', 'everything', 'good', 'is', 'that', 'this']

vectorizer=CountVectorizer()
spamnormal_vectorizer=vectorizer.fit_transform(df_new["text"])
vectorizer.get_feature_names_out()## give the unique words from the data
spamnormal_vectorizer.toarray().shape##(5726--no. of data(text), 37178- unique words from the text))

spam_classifier=MultinomialNB()
label=df_new["spam"].values## output label 0 and 1
spam_classifier.fit(spamnormal_vectorizer,label)

## testing with a sample
test=["Money!!","Discount","Hello the day has come provide me the information.Thanks"] 
test_vectorizer=vectorizer.transform(test)
spam_classifier.predict(test_vectorizer)

### splitting training and testing

X=spamnormal_vectorizer
y=label
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
classifier=MultinomialNB()
classifier.fit(X_train,y_train)
## evaluating the classifier
y_predict_test=spam_classifier.predict(X_test)

cm=confusion_matrix(y_test,y_predict_test)

sns.heatmap(cm,annot=True)
plt.show()

print(classification_report(y_test,y_predict_test))