import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import joblib

df=pd.read_excel('Book1.xlsx')
print(df)

df.rename(columns={'GENDER':'gender' , 'WEIGHT':'weight'}, inplace=True )
print(list(df))

df['gender']= df['gender'].replace('male',1)
df['gender']=df['gender'].replace('female',0)
print(df)

# selection input and output 
x=df[['weight','gender']]
y=df['likeness']
# machine learning algorithm  Create and fit our model 
model = DecisionTreeClassifier().fit(x,y)
# predicition 
mode=model.predict([[38,1]])
print(mode)



# Accuracy now we checking but for this we need train and test  
from sklearn.model_selection import   train_test_split
from sklearn.metrics import accuracy_score
# create a model
x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2  ) 
# fitting the model
model= DecisionTreeClassifier().fit(x_train,y_train)

y_unknown= model.predict(x_test)
print(y_unknown)  


# now we are checking accuracy for accuracy we need y_test actual value and predicted value 
score = accuracy_score(y_test, y_unknown)
print('Accuracy Score is ', score)


# how to save your training model even you right again and again 
model= DecisionTreeClassifier().fit(x,y)
joblib.dump(model,'foofie.joblib')

# how to import/run save model on our data 
load= joblib.load('foofie.joblib')
score= load.score(x,y)
print(score)

from sklearn import tree
model = DecisionTreeClassifier().fit(x,y)
tree.export_graphviz(model, out_file='foodie.dot', feature_names=['weight', 'gender'] , class_names=sorted(y.unique()),label='all', rounded=True, filled=True)
