import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat('trainingdata.mat')
#type(data) = dict
#use data.keys() to check all the keys

X = data['X']
y = data['y']

#exampleX = X[np.random.permutation(len(X))[0:100]]
exampleX = X[0:100]
#randomly choose 100 examples 100*400

reshapeX = np.array([0.0]*40000).reshape(200,200)

#for the ith example
for i in range(len(exampleX)): 
    # the ith example will be in the row m and col n
    m = np.floor(i/10)
    n = i%10
    
    for j in range(len(exampleX[i])):
        
        reshapeX[m*20+j%20][n*20+np.floor(j/20)] = exampleX[i][j]

print(type(reshapeX[0][0]))


fig,ax = plt.subplots()
ax.imshow(reshapeX,cmap=plt.cm.gray,vmin=-1,vmax=1,interpolation='nearest')
plt.show()



