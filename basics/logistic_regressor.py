import os

# import torch, numpy, pandas, matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.model import BaseModel


class LogisticRegressor(BaseModel):

    def __init__(self, n, d, model_name='logistic_regressor', model_path='./', auto_save=False):
        super().__init__(model_name, model_path, auto_save)
        self.N = n
        self.D = d
        self.model = nn.Sequential(
            nn.Linear(D, 1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.auto_load()

    def fit(self, X_train, Y_train, X_test, Y_test, n_epochs=1000):
        train_losses = np.zeros(n_epochs)
        test_losses = np.zeros(n_epochs)

        for it in range(n_epochs):
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            outputs = self.model(X_train)
            loss = self.criterion(outputs, Y_train)

            # save the loss
            train_losses[it] = loss.item()

            # backward and optimize
            loss.backward()
            self.optimizer.step()

            output_test = self.model(X_test)
            loss_test = self.criterion(output_test, Y_test)

            test_losses[it] = loss_test.item()

            # print statistics
            if (it + 1) % 10 == 0:
                print(f'Epoch {it + 1}/{n_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {loss_test.item():.4f}')
        if self.auto_save:
            self.save()
        return train_losses, test_losses

    def score(self, X_train, Y_train, X_test, Y_test):
        with torch.no_grad():
            p_train = self.model(X_train)
            p_train = np.round(p_train.data.numpy())
            train_acc = np.mean(p_train == Y_train.data.numpy())

            p_test = self.model(X_test)
            p_test = np.round(p_test.data.numpy())
            test_acc = np.mean(p_test == Y_test.data.numpy())
            return train_acc, test_acc


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    print(type(data))
    print(data.keys())
    print(data.data.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    N, D = X_train.shape

    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    Y_train = torch.from_numpy(Y_train.astype(np.float32).reshape(-1, 1))
    Y_test = torch.from_numpy(Y_test.astype(np.float32).reshape(-1, 1))

    model = LogisticRegressor(N, D, auto_save=True)
    train_losses, test_losses = model.fit(X_train, Y_train, X_test, Y_test)

    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.show()

    train_acc, test_acc = model.score(X_train, Y_train, X_test, Y_test)
    print(f'Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

