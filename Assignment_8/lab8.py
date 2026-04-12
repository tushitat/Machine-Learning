import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def step(x): return 1 if x >= 0 else 0
def bipolar(x): return 1 if x >= 0 else -1
def sigmoid(x): return 1/(1+np.exp(-x))
def relu(x): return max(0,x)
def tanh(x): return np.tanh(x)

def predict(x, w, act):
    return act(np.dot(x, w))

def train(X, y, w, lr, act):
    errors=[]
    for epoch in range(1000):
        total=0
        for i in range(len(X)):
            y_pred=predict(X[i],w,act)
            e=y[i]-y_pred
            w=w+lr*e*X[i]
            total+=e**2
        errors.append(total)
        if total<=0.002:
            break
    return w, errors, epoch+1

def run_activation_compare(X,y,w,lr):
    acts=[("Step",step),("Bipolar",bipolar),("Sigmoid",sigmoid),("ReLU",relu)]
    res={}
    for name,fn in acts:
        _,_,ep=train(X,y,w.copy(),lr,fn)
        res[name]=ep
    return res

def learning_rate_test(X,y,w):
    rates=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    iters=[]
    for lr in rates:
        _,_,ep=train(X,y,w.copy(),lr,step)
        iters.append(ep)
    return rates,iters

def pseudo_inverse(X,y):
    X_aug=np.c_[np.ones(len(X)),X]
    w=np.linalg.pinv(X_aug.T@X_aug)@X_aug.T@y
    return w

def simple_backprop(X,y,lr):
    w=np.random.rand(3)
    errors=[]
    for epoch in range(1000):
        total=0
        for i in range(len(X)):
            x=X[i]
            o=sigmoid(np.dot(x,w))
            e=y[i]-o
            w=w+lr*e*x
            total+=e**2
        errors.append(total)
        if total<=0.002:
            break
    return w,errors

if __name__=="__main__":

    X=np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    y_and=np.array([0,0,0,1])
    y_xor=np.array([0,1,1,0])

    w=np.array([10.0,0.2,-0.75])
    lr=0.05

    w_final,err,ep=train(X,y_and,w.copy(),lr,step)
    print("A2 AND Epochs:",ep)

    plt.plot(err)
    plt.title("A2 Error")
    plt.show()

    res=run_activation_compare(X,y_and,w.copy(),lr)
    print("A3:",res)

    rates,iters=learning_rate_test(X,y_and,w.copy())
    plt.plot(rates,iters)
    plt.title("A4 Learning Rate vs Iterations")
    plt.show()

    w_xor,err_xor,ep_xor=train(X,y_xor,w.copy(),lr,step)
    print("A5 XOR Epochs:",ep_xor)

    X_cust=np.array([
        [20,6,2,386],[16,3,6,289],[27,6,2,393],[19,1,2,110],[24,4,2,280],
        [22,1,5,167],[15,4,2,271],[18,4,2,274],[21,1,4,148],[16,2,4,198]
    ])
    y_cust=np.array([1,1,1,0,1,0,1,1,0,0])

    X_cust=np.c_[np.ones(len(X_cust)),X_cust]
    w_cust,_ ,_=train(X_cust,y_cust,np.random.rand(X_cust.shape[1]),0.01,sigmoid)
    print("A6 Done")

    w_pi=pseudo_inverse(X_cust[:,1:],y_cust)
    print("A7 Weights:",w_pi)

    w_bp,err_bp=simple_backprop(X,y_and,0.05)
    print("A8 Done")

    w_xor_bp,err_xor_bp=simple_backprop(X,y_xor,0.05)
    print("A9 Done")

    y_two=np.array([[1,0],[1,0],[1,0],[0,1]])
    w2=np.random.rand(3,2)
    for epoch in range(100):
        for i in range(len(X)):
            o=sigmoid(X[i]@w2)
            e=y_two[i]-o
            w2=w2+0.05*np.outer(X[i],e)
    print("A10 Done")

    clf_and=MLPClassifier(max_iter=500)
    clf_and.fit(X,y_and)
    print("A11 AND:",clf_and.predict(X))

    clf_xor=MLPClassifier(max_iter=500)
    clf_xor.fit(X,y_xor)
    print("A11 XOR:",clf_xor.predict(X))