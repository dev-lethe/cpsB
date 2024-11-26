import os
import glob
import numpy as np
import cv2
import argparse
from img2feat import CNN

####
from img2feat import antbee
from sklearn.decomposition import PCA
####
from util import *
####


def main(dir_imgs, network, model_name, epochs):
    epochs = int(epochs)
    # get antbee fueatures by CNN
    net = CNN(network)

    if network == "alexnet":
        n_input = 256
    elif network == "vgg16":
        n_input = 512
    elif network == "resnet152":
        n_input = 2048

    (Xtrain, Ytrain), (Xtest, Ytest) = antbee.load_squared_npy(network)   
    (Xtrain, Ytrain) = DA(Xtrain, Ytrain)

    # def model
    model = NN(n_in=n_input)

    # training
    for epoch in range(1, epochs+1):
        model.forward(Xtrain)
        loss_train = model.loss(Ytrain)

        model.backward()
        model.update()

        ypred = model.forward(Xtest)
        acc = accuracy_np(Ytest, ypred)
        print(f"epoch : {epoch:>2} | train loss : {loss_train:>-3} | test acc : {acc:.2%}")

    # output model
#    model_name = f"NN_{ver}" + ".pkl"
#    save_model(model_name, model)

if(__name__ =="__main__"):
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("-n", "--network", default="alexnet")
    parser.add_argument("-m", "--model_name", default="model")
    parser.add_argument("-d", "--dir_imgs", default="antbee/train/")
    parser.add_argument("-e", "--epochs", default=10)


    args = parser.parse_args()
    main(**vars(args))