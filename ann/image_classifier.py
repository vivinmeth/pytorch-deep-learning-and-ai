import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils.model import BaseModel


class AnnImageClassifier(BaseModel):

    def __init__(self, model_name='ann_image_classifier', model_path='./', optimizer_params=None, auto_save=False):
        super().__init__(model_name, model_path, auto_save)
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        print(f"{self.model_name} is using {self.device}")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        if not optimizer_params:
            optimizer_params = {
                # 'lr': 0.001,
            }
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)

        self.auto_load = True
        self.auto_loader()

    def fit(self, train_dataset, test_dataset, n_epochs=10, batch_size=128):
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        train_losses = np.zeros(n_epochs)
        test_losses = np.zeros(n_epochs)

        for it in range(n_epochs):
            train_loss = []
            test_loss = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                inputs = inputs.view(-1, 784)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)

            train_losses[it] = train_loss

            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                inputs = inputs.view(-1, 784)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss.append(loss.item())
            test_loss = np.mean(test_loss)

            test_losses[it] = test_loss
            print(f'Epoch {it + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if self.auto_save:
            self.save()
        return train_losses, test_losses


    def score(self, train_dataset, test_dataset):
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

        n_train_correct = 0
        n_train_total = 0
        n_test_correct = 0
        n_test_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            inputs = inputs.view(-1, 784)

            outputs = self.model(inputs)
            _, predictions = torch.max(outputs, 1)
            n_train_correct += (predictions == targets).sum().item()
            n_train_total += targets.shape[0]

        train_acc = n_train_correct / n_train_total

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            inputs = inputs.view(-1, 784)

            outputs = self.model(inputs)
            _, predictions = torch.max(outputs, 1)
            n_test_correct += (predictions == targets).sum().item()
            n_test_total += targets.shape[0]

        test_acc = n_test_correct / n_test_total

        print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
        return train_acc, test_acc




if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from torchvision import transforms

    train_dataset = MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataset = MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    X_train = train_dataset.data
    Y_train = train_dataset.targets

    X_test = test_dataset.data
    Y_test = test_dataset.targets

    model = AnnImageClassifier(auto_save=True)
    train_losses, test_losses = model.fit(train_dataset, test_dataset, n_epochs=10)
    model.score(train_dataset, test_dataset)

    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.show()



