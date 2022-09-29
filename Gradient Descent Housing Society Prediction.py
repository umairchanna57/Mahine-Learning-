from cmath import cos
import re
from tkinter.messagebox import RETRY
import numpy as np 
import copy
import math
X_train=np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
y_train=np.array([460,232,178])
print(f'X_train : {X_train} , Shape is {X_train.shape}  ')
print(f'Y_train : {y_train} , shape is {y_train.shape}')

# Parameters 
# Scaler
b_init = 785.1811367994083
# 𝐰  is a vector with  𝑛  elements
w_init=np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f'w is {w_init} , Shape is : {w_init.shape}')
print(f'B is Scaler : {b_init}')


def predict(x, w,b):
 p=0
 n=x.shape[0]
 for i in range(n):
    p_i=x[i]*w[i]
    p=p+p_i
 p=p+b
 return p


X=X_train[0,:]
print('Housing  Prediction By Single loop ',predict(X,w_init,b_init))



def multi_prediction(x,w,b):
  p=np.dot(x,w)+b
  return p 

print('Over all Housing prediction', multi_prediction(X_train,w_init,b_init))



def cost(x,y,w,b): 
    m=x.shape[0]
    cost=0.0
    for i in range (m):
        f_wi=np.dot(x[i], w)+b
        cost=cost+(f_wi-y[i])**2
    cost=cost/(2*m)
    return cost
print(f'Cost is : {cost(X_train,y_train,w_init,b_init)}')

# def gradient(x,y,w,b):
   
#    m,n=x.shape
#    dj_dw = np.zeros((n,))
#    dj_db=0.
    
#    for i in range(m):
#         err=(np.dot(x[i],w)+b)-y[i]
#         for j in range(n):
#             dj_dw[j] = dj_dw[j] + err* x[i, j]       
#         dj_db=dj_db+err
#    dj_dw=dj_dw/m
#    dj_db=dj_db/m
   
#    return dj_dw,dj_db
def gradient(X, y, w, b): 
   
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err* X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw




tmp_dj_db, tmp_dj_dw = gradient(X_train, y_train, w_init, b_init)
print(tmp_dj_db)
print(tmp_dj_dw)


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
   
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
        
    return w, b, J_history #return final w,b and J history for graphing
    # initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                   cost,gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
