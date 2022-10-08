import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# import data 
df=pd.read_csv('Salary_Data.csv')
print(df.head() )

# splitting dataset into training data and testing data 

x=df[['YearsExperience']]
y=df['Salary']

# split data 

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# fit linear regression 
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train,y_train)
print(model)

print(model.predict([[5],[3],[20],[6]]))
print(model.score(x_test,y_test))
plt.scatter(x_test,y_test)
plt.plot(x_train,model.predict(x_train))
plt.show()


reg = LinearRegression().fit(x_test,y_test)
print('regression is ', reg.score(x_test,y_test))
