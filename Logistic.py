import os
import glob
import numpy as np
import cv2
import argparse
from img2feat import CNN
from sklearn.linear_model import LogisticRegression

####
from img2feat import antbee
####
from util import accuracy
####

def regression(X, Y):
    model = LogisticRegression()
    model.fit(X, Y)
    return model

def main(dir_imgs, network, model_name):
    net = CNN(network)
    # (Train, ytrain), (Test, ytest) = antbee.load()

    (Xtrain, Ytrain), (Xtest, Ytest) = antbee.load_squared_npy(network)
    
    #Train = np.array(Train)
    #Test = np.array(Test)
    #Ytrain = np.array(Ytrain)
    #Ytest = np.array(Ytest)

    #Xtrain = net([Train])
    #Xtest = net([Test])

    model = regression(Xtrain, Ytrain)

    ypred = model.predict(Xtest)
    acc = accuracy(Ytest, ypred)
    print("acc :", acc)

if(__name__ =="__main__"):
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("-n", "--network", default="alexnet")
    parser.add_argument("-m", "--model_name", default="model")
    parser.add_argument("-d", "--dir_imgs", default="antbee/train/")


    args = parser.parse_args()
    main(**vars(args))