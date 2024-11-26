import os
import glob
import numpy as np
import cv2
import argparse
from img2feat import CNN
from sklearn.neighbors import KNeighborsClassifier

####
from img2feat import antbee
####
from util import accuracy
####

def regression(X, Y, k):
    model = KNeighborsClassifier(k)
    model.fit(X, Y)
    return model

def main(dir_imgs, network, model_name, k_neighbor):
    k = int(k_neighbor)
    net = CNN(network)
    #(Train, Ytrain), (Test, Ytest) = antbee.load()
    (Xtrain, Ytrain), (Xtest, Ytest) = antbee.load_squared_npy(network)
    
    #Train = np.array(Train)
    #Test = np.array(Test)
    #Ytrain = np.array(Ytrain)
    #Ytest = np.array(Ytest)

    #Xtrain = net([Train])
    #Xtest = net([Test])

    ylist = []
    for i in range(1, k+1):
        model = regression(Xtrain, Ytrain, i)
        ypred = model.predict(Xtest)
        ylist.append(ypred)
    ylist = np.array(ylist)
    ypred = np.mean(ylist, axis=0, dtype=int)

    acc = accuracy(Ytest, ypred)
    print("test acc :", acc)

if(__name__ =="__main__"):
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("-n", "--network", default="alexnet")
    parser.add_argument("-m", "--model_name", default="model")
    parser.add_argument("-d", "--dir_imgs", default="antbee/train/")

    parser.add_argument("-k", "--k_neighbor", default=3, required=True)


    args = parser.parse_args()
    main(**vars(args))