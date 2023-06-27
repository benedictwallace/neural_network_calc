import numpy as np
import pandas as pd
import sys
#np.set_printoptions(threshold=sys.maxsize)

input_no = 20
hidden_no = 50
output_no = 201

# Training data in the form [(a)(b)(op)(c)] for a op b = c, a and b are 8-bit binary, c denary
data = pd.read_csv("C:/Users/b.wallace/Desktop/python/input_data/training_data.csv", header=None, index_col=False)

data = np.array(data)

m, n = data.shape

np.random.shuffle(data)

data_dev = data[1:1000].T
Y_dev = data_dev[input_no]
X_dev = data_dev[0:input_no]

data_train = data[1000:m].T
Y_train = data_train[input_no]
X_train = data_train[0:input_no]

# Initialise
def initial():
    weights_1 = np.random.rand(hidden_no, input_no) - 0.5
    
    weights_2 = np.random.rand(output_no, hidden_no) - 0.5
    
    bias_1 = np.random.rand(hidden_no, 1) - 0.5
    
    bias_2 = np.random.rand(output_no, 1) - 0.5
    
    return weights_1, weights_2, bias_1, bias_2
#-----------------

def dtb(n):
    # returns an 8-bit binary string of int n
    n = int(n)
    a = str(bin(n).replace("0b", ""))
    while len(a) < 8:
        a = "0" + a
    return a

def sig(x):
    # Sigma function caps values incase of overflow
    x = np.clip(x, -500, 500)
    return 1.0/(1.0 + np.exp(-x))

def sig_dash(x):
    # Derivative of sigma function caps values incase of overflow
    x = np.clip(x, -500, 500)
    return np.exp(-x)/(np.square(1.0 + np.exp(-x)))

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_dash(x):
    # Derivative of RelU function
    return x > 0

def softmax(z):
    return np.exp(z)/sum(np.exp(z))

# Takes in list of correct answers [3,5,...] outputs matrix of ones in these positions
def format_out(y):
    ym = np.zeros((output_no, y.size))
    for i in range(0, y.size):
    
        ym[y[i]][i] = 1
    return ym

def get_predictions(a_2):
    return np.argmax(a_2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def forward(weights_1, weights_2, bias_1, bias_2, x):
    
    z_1 = weights_1.dot(x) + bias_1
   
    a_1 = ReLU(z_1)
    
    z_2 = weights_2.dot(a_1) + bias_2
 
    a_2 = softmax(z_2)
    
    return z_1, a_1, z_2, a_2

def backward(z_1, a_1, a_2, weights_2, x, y):
    ym = format_out(y)
    
    dz_2 = a_2 - ym
    dw_2 = 1/m * dz_2.dot(a_1.T)
    db_2 = 1/m * np.sum(dz_2)
    
    dz_1 = np.multiply(weights_2.T.dot(dz_2), ReLU_dash(z_1))
    dw_1 = 1/m * dz_1.dot(x.T)
    db_1 = 1/m * np.sum(dz_1)

    return dw_1, db_1, dw_2, db_2

def update(weights_1, weights_2, bias_1, bias_2, dw_1, db_1, dw_2, db_2, alpha):
    weights_1 = weights_1 - alpha*dw_1
    weights_2 = weights_2 - alpha*dw_2

    db_1 = db_1.reshape(-1,1)
    db_2 = db_2.reshape(-1,1)

    bias_1 = bias_1 - alpha*db_1
    bias_2 = bias_2 - alpha*db_2
    
    return weights_1, weights_2, bias_1, bias_2

def gradient_decent(x, y, iterations, alpha):

    weights_1, weights_2, bias_1, bias_2 = initial()
    for i in range(iterations):
        z_1, a_1, z_2, a_2 = forward(weights_1, weights_2, bias_1, bias_2, x)
        dw_1, db_1, dw_2, db_2 = backward(z_1, a_1, a_2, weights_2, x, y)
        weights_1, weights_2, bias_1, bias_2 = update(weights_1, weights_2, bias_1, bias_2, dw_1, db_1, dw_2, db_2, alpha)
        
        if i % 10 == 0:
            print("iteration: ", i)
            predictions = get_predictions(a_2)
            print(get_accuracy(predictions, y))

    return weights_1, weights_2, bias_1, bias_2

"""
# Comment out this section if wanting to use pre-found values ------------------------
weights_1, weights_2, bias_1, bias_2 = gradient_decent(X_train, Y_train, 100000, 0.1)

dfw1 = pd.DataFrame(weights_1)
dfw1.to_csv("C:/Users/b.wallace/Desktop/python/input_data/weights_1.csv", header=False, index=False)

dfw2 = pd.DataFrame(weights_2)
dfw2.to_csv("C:/Users/b.wallace/Desktop/python/input_data/weights_2.csv", header=False, index=False)

dfb1 = pd.DataFrame(bias_1)
dfb1.to_csv("C:/Users/b.wallace/Desktop/python/input_data/bias_1.csv", header=False, index=False)

dfb2 = pd.DataFrame(bias_2)
dfb2.to_csv("C:/Users/b.wallace/Desktop/python/input_data/bias_2.csv", header=False, index=False)
# ------------------------------------------------------------------------------------
"""

# Comment out this section if wanting to find weights and bias's ---------------------

weights_1 = np.array(pd.read_csv("C:/Users/b.wallace/Desktop/python/input_data/weights_1.csv", header=None, index_col=False))
weights_2 = np.array(pd.read_csv("C:/Users/b.wallace/Desktop/python/input_data/weights_2.csv", header=None, index_col=False))
bias_1 = np.array(pd.read_csv("C:/Users/b.wallace/Desktop/python/input_data/bias_1.csv", header=None, index_col=False))
bias_2 = np.array(pd.read_csv("C:/Users/b.wallace/Desktop/python/input_data/bias_2.csv", header=None, index_col=False))

# ------------------------------------------------------------------------------------
while True:
    print("Numbers are of form 200 > x > y > 0, make sure the answer is less than 200 and an integer")
    x = input("Enter the first number (x): ")
    op = input("Enter the operation (+,-,*,/): ")
    y = input("Enter the second number (y): ")
    f = dtb(x) + dtb(y)
    
    if op == "-":
        f += "1000"
    elif op == "+":
        f += "0100"
    elif op == "*":
        f += "0010"
    elif op == "/":
        f += "0001"
    f = [*f]
    
    f = [int(x) for x in f]
    f = pd.DataFrame(f)
    f = np.array(f)
   
    z1,a1,z2,a2 = forward(weights_1, weights_2, bias_1, bias_2, f)
    print(x, " ", op, " ", y, " = ", get_predictions(a2)[0])
