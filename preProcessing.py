'''
tool for pre-processing
scaling: compute the mean value and standard deviation in each column of X
         then sacle X by first subtracting the mean value second dividing by std
         in their corresponding column
'''

import numpy as np

def scaling(X):
    scaledX = X.copy()

    x_mean = np.mean(X,axis=0)
    #mean of each column in X, use axis=1 to compute row
    x_std = np.std(scaledX,axis=0)
    #standard deviation of each column in X, use axis=1 to compute row

    for row in scaledX:
        for i in range(len(row)):
            if x_std[i] == 0:
                row[i] = 1
            else:
                row[i] = float((row[i]-x_mean[i])/x_std[i])
    return (scaledX,x_mean,x_std)

if __name__ == '__main__':
    X = np.array([[1,2,3,4],[5,6,7,8]])
    scaledX, x_mean, x_std = scaling(X)
    print(scaledX,x_mean,x_std)
