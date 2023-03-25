import os.path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class LinearRegressor:

    def __init__(self, model_name='linear_regressor', model_path='./', optimizer_params=None, auto_save=False):
        self.auto_save = auto_save
        self.model_name = model_name
        self.model_path = os.path.abspath(f"{model_path}/{model_name}.model")
        self.model = nn.Linear(1, 1)
        self.criterion = nn.MSELoss()

        if not optimizer_params:
            optimizer_params = {
                'lr': 0.1,
            }
        self.optimizer = torch.optim.SGD(self.model.parameters(), **optimizer_params)
        if os.path.exists(self.model_path):
            print(f'model found at {self.model_path}')
            self.load()

    def weight(self):
        return self.model.weight.data.numpy()

    def bias(self):
        return self.model.bias.data.numpy()

    def load(self):
        print("Loading model...")
        self.model.load_state_dict(torch.load(self.model_path))

    def save(self):
        print("Saving model...")
        torch.save(self.model.state_dict(), self.model_path)

    def fit(self, X, Y, n_epochs=30):
        inputs = torch.from_numpy(X.astype(np.float32))
        targets = torch.from_numpy(Y.astype(np.float32))

        losses = []
        outputs = None

        print(f"Training {self.model_name} for {n_epochs} epochs...")
        for it in range(n_epochs):
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # save the loss
            losses.append(loss.item())

            # backward and optimize
            loss.backward()
            self.optimizer.step()

            print(f'Epoch {it + 1}/{n_epochs}, Loss: {loss.item():.4f}')
        print(f"Training {self.model_name} complete!")
        if self.auto_save:
            self.save()
        return inputs, outputs, losses

    def predict(self, X):
        inputs = torch.from_numpy(X.astype(np.float32))
        outputs = self.model(inputs)
        return outputs.detach().numpy()

    def score(self, X, Y):
        inputs = torch.from_numpy(X.astype(np.float32))
        targets = torch.from_numpy(Y.astype(np.float32))
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return loss.item()

    # Functional way
    @classmethod
    def train_linear_model(cls, X, Y, optimizer_params=None, n_epochs=30):
        model = nn.Linear(1, 1)

        criterion = nn.MSELoss()

        if not optimizer_params:
            optimizer_params = {
                'lr': 0.1,
            }

        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)

        inputs = torch.from_numpy(X.astype(np.float32))
        targets = torch.from_numpy(Y.astype(np.float32))

        type(inputs)
        type(targets)  # torch.Tensor, torch.Tensor

        losses = []
        outputs = None

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

            print(f'Epoch {it + 1}/{n_epochs}, Loss: {loss.item():.4f}')
        return model, inputs, outputs, losses


if __name__ == '__main__':
    # we would like to generate 20 data points
    N = 20

    # random data on the x-azis in (-5, +5)
    X = np.random.random(N) * 10 - 5

    # a line plus some noise
    Y = 0.5 * X + 1 + np.random.randn(N)


    plt.scatter(X, Y)


    X = X.reshape(N, 1)
    Y = Y.reshape(N, 1)
    model = LinearRegressor(auto_save=True)

    inputs, outputs, losses = model.fit(X, Y, n_epochs=30)

    plt.plot(losses)

    predicted = model.predict(X)
    plt.scatter(X, Y, label='Original data', s=25)
    plt.plot(X, predicted, label='Fitted line')
    plt.legend()
    plt.show()


    # another way to use model
    with torch.no_grad():
        predicted = model.predict(X)
    print(predicted)


    # Model efficiency
    w = model.weight()
    b = model.bias()
    print(w, b)