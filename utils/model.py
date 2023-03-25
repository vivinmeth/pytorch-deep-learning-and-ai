import os

import numpy as np
import torch


class BaseModel:
    auto_load = False
    def __init__(self, model_name='torch_base', model_path='./', auto_save=False):
        self.auto_save = auto_save
        self.model_name = model_name
        self.model_path = os.path.abspath(f"{model_path}/{model_name}.model")
        self.model = None
        self.criterion = None
        self.optimizer = None

    def weight(self):
        return self.model.weight.data.numpy()

    def bias(self):
        return self.model.bias.data.numpy()

    def auto_loader(self):
        if self.auto_load and os.path.exists(self.model_path):
            print(f'model found at {self.model_path}')
            self.load()

    def load(self):
        print("Loading model...")
        self.model.load_state_dict(torch.load(self.model_path))

    def save(self):
        print("Saving model...")
        torch.save(self.model.state_dict(), self.model_path)

    def fit(self):
        pass

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
