import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from layer_classes import FlattenCfg, LinearCfg
from optimizer import ABCOptimizer, TransformerOptimizer

test = pd.read_csv("data/churn/test.csv")
train = pd.read_csv("data/churn/train.csv")

train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce').fillna(0)

y_train = train['Churn'].map({'Yes': 1, 'No': 0})
X_train = train[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']]

X_train = pd.get_dummies(X_train, drop_first=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

initial_arch = [
    FlattenCfg(),
    LinearCfg(in_features=0, out_features=32, activation=nn.ReLU),
    LinearCfg(in_features=0, out_features=1, activation=None)
]

opt_trans = TransformerOptimizer(layers=initial_arch, max_layers=50, dataset=train_loader)
best_arch_trans, stats_trans = opt_trans.run(50)

opt_abc = ABCOptimizer(layers=best_arch_trans, dataset=train_loader)
best_sol_final, optim_stats_abc = opt_abc.run(20)