import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

'''
given a 2 dimentional array
represent them as gray scale image
'''

def representData(X,image_width):

    X = np.array(X)
    m,n = X.shape
    #m: training sample number
    #n: training sample dimention

    image_heigh = n/image_width

    display_cols = np.floor(np.sqrt(m))
    display_rows = np.ceil(m/display_cols)

    reshapeX = np.zeros((image_heigh*display_rows,image_width*display_cols))

    #for the ith example
    for i in range(m):
        # the ith example will be in the row p and col q
        p = np.floor(i/display_cols)
        q = i%display_cols

        for j in range(n):
            #x[i] will be reshaped vertically
            reshapeX[p*image_heigh+j%image_heigh]\
                    [q*image_width+np.floor(j/image_width)] = X[i][j]

    fig,ax = plt.subplots()
    ax.imshow(reshapeX,cmap=plt.cm.gray,vmin=-1,vmax=1,interpolation='nearest')
    plt.show()


if __name__ == '__main__':

    data = sio.loadmat('trainingdata.mat')
    #type(data) = dict
    #use data.keys() to check all the keys

    X = data['X']
    y = data['y']
    #type(X) = ndarray

    exampleX = X[np.random.permutation(len(X))[0:100]]
    #randomly choose some of the data as example

    representData(exampleX,20)
