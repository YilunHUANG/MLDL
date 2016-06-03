import numpy as np
import preProcessing
import matplotlib.pyplot as plt

def hypo(X,Theta):
    #the function to predict the final answer
    #x can be a vector or matrix
    #Theta from theta_0 to theta_n
    X = np.array(X)
    Theta = np.array(Theta)
    #we need to add one column to X because of theta_0
    return np.dot(X,Theta)

def prCost(X,Theta,lambda_,Y):
    #given the current Theta, compute the cost
    #including the regularization term
    #(no penalty for theta_0)
    m = X.shape[0]
    return np.sum(np.power(Y-hypo(X,Theta),2))/(2*m)+\
           np.sum(np.power(Theta[1:],2))*lambda_/(2*m)

def gradient(X,Theta,lambda_,Y):
    #given the current Theta compute grade of Theta
    #grad(theta_0) = (1/m)*sum_i=1_m((hypo(x_i)-y_i)^2*x_0_i)
    X = np.array(X)
    m = X.shape[0]
    grad = np.dot(X.T,hypo(X,Theta)-Y)/m+\
           lambda_*Theta/m
    grad[0] = grad[0] - lambda_*Theta[0]/m
    return grad

def GD(X,Theta,lambda_,alpha,Y):
    #gradient descent
    turn = 0
    grad = gradient(X,Theta,lambda_,Y)
    while np.max(np.absolute(grad))>0.001:
        Theta = Theta - alpha*grad
        grad = gradient(X,Theta,lambda_,Y)
        turn += 1
    print("used: ",turn," turns to train, finla cost: ",prCost(X,Theta,lambda_,Y))
    return Theta

if __name__ == '__main__':
    #preparing training example
    #using sin and random number
    X = np.linspace(0,2*np.pi,30)
    Y = np.sin(X) + np.random.uniform(-0.2,0.2,X.shape[0])

    #plot the example and the sin function
    plt.scatter(X,Y)
    sinX = np.linspace(0,2*np.pi,100)
    sinY = np.sin(sinX)
    plt.plot(sinX,sinY)

    #expanding X to its exponentiation
    #eg. x becomes x^1 x^2 x^3
    #then add a column of one to X: 1 x^1 x^2 x^3
    exp = 4
    x_exp = np.zeros((X.size,exp))
    for i in range(exp):
        x_exp[:,i] = np.power(X,i+1)
    x_exp = np.hstack((np.ones((x_exp.shape[0],1)),x_exp))

    #scaling the training example
    x_exp_scaled,x_mean,x_std = preProcessing.scaling(x_exp)
    #initializing Theta and training it using GD
    Theta = np.random.uniform(-0.5,0.5,(exp+1))
    #regularization parameter
    lambda_ = 0.01
    #learning rate
    alpha = 0.3
    #use GD to train the Theta
    Theta = GD(x_exp_scaled,Theta,lambda_,alpha,Y)

    #scale the sinX using previous x_mean and x_std
    #then compute predicted result predY and draw the line
    print("Theta: ",Theta)
    sinX_exp = np.zeros((sinX.size,exp))
    for i in range(exp):
        sinX_exp[:,i] = np.power(sinX,i+1)
    sinX_exp = np.hstack((np.ones((sinX_exp.shape[0],1)),sinX_exp))
    for row in sinX_exp:
        for i in range(len(row)):
            if x_std[i] == 0:
                row[i] = 1
            else:
                row[i] = float((row[i]-x_mean[i])/x_std[i])
    predY = hypo(sinX_exp,Theta)
    plt.plot(sinX,predY)
    plt.legend(["sin(x)","hypo(x)"])

    plt.show()
