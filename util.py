import numpy as np
from sklearn.decomposition import PCA

def DA(X, Y):
    X_lr = np.fliplr(X).copy()
    X_ud = np.flipud(X).copy()
    X_udlr = np.flipud(X_lr).copy()
    X = np.concatenate([X, X_lr, X_ud, X_udlr])
    Y_add = np.copy(Y)
    Y = np.concatenate([Y, Y_add, Y_add, Y_add])
    return X, np.eye(np.max(Y) + 1)[Y]

def accuracy(ans, pred):
    num = len(ans)
    true = 0
    total = 0
    for i in range(num):
        if abs(ans[i] - pred[i]) < 0.5:
            true += 1
        total += 1
    acc = true / total
    return acc

def accuracy_np(ans, pred):
    ans_class = ans
    pred_class = np.argmax(pred, axis=1)
    
    correct = np.sum(ans_class == pred_class)
    total = len(ans_class)
    
    acc = correct / total
    return acc

"""
class ReLU_np():
    def __call__(self, x):
        self.u = (x <= 0)
        return np.where(self.u, 0, x)

    def bp(self, x):
        return np.where(self.u, 0, x)
"""
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def init_weights(M, L, std):
    return np.random.randn(M, L) * std, np.zeros(M)
"""
class ReLULayer():
    def __init__(self, n_inputs, n_units, std=0.01):
        self.w, self.b = init_weights(n_units,n_inputs,std)
        self.relu = ReLU_np()

    def __call__(self, x):
        self.x = x
        self.u = np.dot(x, self.w.T) + self.b
        return self.relu(self.u)

    def backward(self, d):
        delta = self.relu.bp(d)
        self.dw = np.dot(delta.T, self.x)
        self.db = np.sum(delta, axis=0)
        return np.dot(delta, self.w)

    def update(self, eta=0.5):
        self.w -= eta * self.dw
        self.b -= eta * self.db
"""
"""
class SoftmaxLayer():
    def __init__(self, n_inputs, n_units, std=0.01):
        self.w, self.b = init_weights(n_units,n_inputs,std)

    def __call__(self, x):
        self.x = x
        self.u = np.dot(x, self.w.T) + self.b
        self.z = softmax(self.u)
        return self.z

    def cee(self, y):
        self.y = y
        return -np.sum(y * np.log(self.z + 1e-8)) / y.shape[0]

    def backward(self):
        delta = (self.z - self.y) / self.y.shape[0]
        self.dw = np.dot(delta.T, self.x)
        self.db = np.sum(delta, axis=0)
        return np.dot(delta, self.w)

    def update(self, eta=0.05):
        self.w -= eta * self.dw
        self.b -= eta * self.db
"""
class ReLU_np():
    def __call__(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def bp(self, dout):
        return dout * self.mask

class ReLULayer():
    def __init__(self, n_inputs, n_units, std=0.01):
        self.w, self.b = init_weights(n_units, n_inputs, std)
        self.relu = ReLU_np()

    def __call__(self, x):
        self.x = x
        self.u = np.dot(x, self.w.T) + self.b
        return self.relu(self.u)

    def backward(self, dout):
        delta = self.relu.bp(dout)
        self.dw = np.dot(delta.T, self.x)
        self.db = np.sum(delta, axis=0)
        return np.dot(delta, self.w)

    def update(self, eta=0.001):
        self.w -= eta * self.dw
        self.b -= eta * self.db

class SoftmaxLayer():
    def __init__(self, n_inputs, n_units, std=0.01):
        self.w, self.b = init_weights(n_units, n_inputs, std)

    def __call__(self, x):
        self.x = x
        self.u = np.dot(x, self.w.T) + self.b
        self.z = softmax(self.u)
        return self.z

    def cee(self, y):
        self.y = y
        return -np.sum(y * np.log(self.z + 1e-8)) / y.shape[0]

    def backward(self):
        delta = (self.z - self.y) / self.y.shape[0]
        self.dw = np.dot(delta.T, self.x)
        self.db = np.sum(delta, axis=0)
        return np.dot(delta, self.w)

    def update(self, eta=0.001):
        self.w -= eta * self.dw
        self.b -= eta * self.db


class NN():
    def __init__(self, n_in=256, n_out=2):
        o_1 = 1048
        o_2 = 256
        o_3 = 256
        o_4 = 64

        self.l1 = ReLULayer(n_in, o_1)
        self.l2 = ReLULayer(o_1, o_2)
        self.l3 = ReLULayer(o_2, o_3)
        self.l4 = ReLULayer(o_3, o_4)
        self.out = SoftmaxLayer(o_4, n_out)

        self.PCA = PCA(o_1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.out(x)

        return x
    
    def forward_PCA(self, x):
        x = self.PCA().fit(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.out(x)

        return x
    
    def loss(self, y):
        return self.out.cee(y)
    
    def backward(self):
        h = self.out.backward()
        h = self.l4.backward(h)
        h = self.l3.backward(h)
        h = self.l2.backward(h)
        h = self.l1.backward(h)
    
    def update(self, lr=0.001):
        self.l1.update(lr)
        self.l2.update(lr)
        self.l3.update(lr)
        self.l4.update(lr)
        self.out.update(lr)