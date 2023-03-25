import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# we would like to generate 20 data points
N = 20

# random data on the x-azis in (-5, +5)
X = np.random.random(N) * 10 - 5

# a line plus some noise
Y = 0.5 * X + 1 + np.random.randn(N)


plt.scatter(X, Y)


model = nn.Linear(1, 1)


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


X = X.reshape(N, 1)
Y = Y.reshape(N, 1)

inputs = torch.from_numpy(X.astype(np.float32))
targets = torch.from_numpy(Y.astype(np.float32))

type(inputs)
type(targets)  # torch.Tensor, torch.Tensor


n_epochs = 30
losses = []

for it in range(n_epochs):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # save the loss
    losses.append(loss.item())

    # backward and optimize
    loss.backward()
    optimizer.step()

    print(f'Epoch {it+1}/{n_epochs}, Loss: {loss.item():.4f}')

# Way 1
predicted = model(inputs).detach().numpy()
plt.scatter(X, Y, label='Original data', s=25)
plt.plot(X, predicted, label='Fitted line')
plt.legend()
plt.show()


# another way to use model
with torch.no_grad():
    predicted = model(inputs).numpy()
print(predicted)



# Model efficiency
w = model.weight.data.numpy()
b = model.bias.data.numpy()
print(w, b)

