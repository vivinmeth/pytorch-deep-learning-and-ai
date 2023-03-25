import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils.model import BaseModel


class AnnRegressor(BaseModel):

    def __init__(self, model_name='ann_regressor', model_path='./', optimizer_params=None, auto_save=False):
        super().__init__(model_name, model_path, auto_save)
        self.model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        print(f"{self.model_name} is using {self.device}")
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        if not optimizer_params:
            optimizer_params = {
                'lr': 0.01,
            }
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)

        self.auto_load = True
        self.auto_loader()