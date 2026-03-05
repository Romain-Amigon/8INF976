import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from layer_classes import FlattenCfg, LinearCfg
from optimizer import ABCOptimizer

test = pd.read_csv("C:/Users/amigo/Downloads/test.csv")
train = pd.read_csv("C:/Users/amigo/Downloads/train.csv")

train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce').fillna(0)

y_train = train['Churn']
X_train = train[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']]

X_train = pd.get_dummies(X_train, drop_first=True)

num_features = X_train.shape[1]

X_train_tensor = torch.tensor(X_train.values.astype(np.float32), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.map({'Yes': 1, 'No': 0}).values, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

layers = []
layers.append(FlattenCfg())
layers.append(LinearCfg(in_features=num_features, out_features=32, activation=nn.ReLU))
layers.append(LinearCfg(in_features=0, out_features=1, activation=None))

opt = ABCOptimizer(
    layers=copy.deepcopy(layers),
    dataset=train_loader,
)

best_sol, optim_stats = opt.run(20)