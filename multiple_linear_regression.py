import pandas as pd 
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('salary_predict_dataset.csv')


print(df.head())

x=df[['experience', 'test_score', 'interview_score']]
y=df['Salary']

print(x)


x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size=0.2 , random_state=0)
model = LinearRegression().fit(x_train,y_train)
print(model)



y_pred= model.predict(x_test)
print( 'Predict test',y_pred)

pred=model.predict([[4,8.0,8.0]])
print('New prediction on the basis of our requirments ',pred)


# coeffient that are in formula 
print(model.coef_)



# reg= LinearRegression().fit(x_test, y_test)
# print('score is ', reg.score(x_test,y_test))
# print('X_train score is ',reg.score(x_train,y_train))




from sklearn.metrics import r2_score
print('score is ',r2_score(y_test, y_pred))




plt.scatter(y_test,y_pred)  
plt.show()