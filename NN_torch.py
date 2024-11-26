import os
import glob
import numpy as np
import cv2
import argparse
from img2feat import CNN

####
import torch
import torch.nn as nn
from img2feat import antbee
from sklearn.decomposition import PCA
####
from util import *
####

def main(dir_imgs, network, model_name, epochs):
    network = "alexnet"
    # get antbee fueatures by CNN
    net = CNN(network)

    (Xtrain, Ytrain), (Xtest, Ytest) = antbee.load_squared_npy(network)
    Xtrain = net(Xtrain)
    Xtest = net(Xtest)

    Xtrain = torch.from_numpy(Xtrain).clone()
    Ytrain = torch.from_numpy(Ytrain).clone()
    Xtest = torch.from_numpy(Xtest).clone()
    Ytest = torch.from_numpy(Ytest).clone()

    # def model
    model = NN()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=0.001, 
                                 betas=(0.9, 0.999), 
                                 eps=1e-08, 
                                 weight_decay=0)

    # training
    for epoch in range(epochs):
        ypred_train = model(Xtrain)

        Ypred = model(Xtest)

        acc = accuracy(Ytest, Ypred)
        print("acc", acc, f"epoch : {epoch:e2}")



    # output model
#    model_name = f"NN_{ver}" + ".pkl"
#    save_model(model_name, model)

if(__name__ =="__main__"):
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("-n", "--network", default="alexnet")
    parser.add_argument("-m", "--model_name", default="model")
    parser.add_argument("-d", "--dir_imgs", default="antbee/train/")
    parser.add_argument("-e", "--epochs", default=10)    # epochs


    args = parser.parse_args()
    main(**vars(args))