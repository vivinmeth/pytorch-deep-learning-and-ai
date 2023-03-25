import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from basics.regression import train_linear_model

# get the data
if os.path.isfile('moore.csv'):
    print('Moore.csv already exists')
else:
    print('Downloading moore.csv...')
    os.system("wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv")


# load the data
data = pd.read_csv('moore.csv', header=None).values
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)

plt.scatter(X, Y)
plt.show()


# since we want a linear modle let's take the log of Y
Y = np.log(Y)
plt.scatter(X, Y)
plt.show()


# let's normalize the data
mx = X.mean()
sx = X.std()
my = Y.mean()
sy = Y.std()
X = (X - mx) / sx
Y = (Y - my) / sy

plt.scatter(X, Y)
plt.show()

# cast to float32
X = X.astype(np.float32)
Y = Y.astype(np.float32)


model, inputs, outputs, losses = train_linear_model(X, Y, optimizer_params={'lr': 0.1, 'momentum': 0.7}, n_epochs=100)

plt.plot(losses)
plt.show()

predicted = model(inputs).detach().numpy()
plt.scatter(X, Y, label='Original data', s=25)
plt.plot(X, predicted, label='Fitted line')
plt.legend()
plt.show()

w = model.weight.data.numpy()
print(w)

