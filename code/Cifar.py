# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 03:26:21 2026

@author: Romain
"""

import types
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import classification_report
from layer_classes import Conv2dCfg, MaxPool2dCfg, FlattenCfg, LinearCfg, DropoutCfg
from optimizer import ABCOptimizer, TransformerOptimizer
from model import DynamicNet

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train_proxy = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_train_final = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

full_trainset_proxy = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_proxy)
full_trainset_final = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_final)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

proxy_size = int(0.3 * len(full_trainset_proxy))
indices = np.random.choice(len(full_trainset_proxy), proxy_size, replace=False)
proxy_dataset = Subset(full_trainset_proxy, indices)

train_loader_proxy = DataLoader(proxy_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
train_loader_final = DataLoader(full_trainset_final, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

def evaluate_cifar_proxy(self, genome, train_epochs=3):
    try:
        model = DynamicNet(genome, input_shape=(3, 32, 32))
        model.to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(train_epochs):
            for inputs, targets in train_loader_proxy:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        return 100. * correct / total
    except Exception:
        return -float('inf')

initial_arch = [
    Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU),
    MaxPool2dCfg(kernel_size=2, stride=2, padding=0),
    FlattenCfg(),
    LinearCfg(in_features=0, out_features=128, activation=nn.ReLU),
    LinearCfg(in_features=0, out_features=10, activation=None)
]

opt_trans = TransformerOptimizer(layers=initial_arch, max_layers=30, dataset=train_loader_proxy, entropy_fct="default")
opt_trans.evaluate = types.MethodType(evaluate_cifar_proxy, opt_trans)

print("Début de la recherche Transformer sur proxy CIFAR-10...")
best_arch_trans, stats_trans = opt_trans.run(30)

opt_abc = ABCOptimizer(layers=best_arch_trans, dataset=train_loader_proxy, limit=5, patience=10)
opt_abc.evaluate = types.MethodType(evaluate_cifar_proxy, opt_abc)

print("\nDébut de l'affinage ABC sur proxy CIFAR-10...")
best_sol_final, optim_stats_abc = opt_abc.run(30)

print("\nEntraînement final de l'architecture optimale sur 100% de CIFAR-10...")
final_model = DynamicNet(best_sol_final, input_shape=(3, 32, 32)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

print("\nEntraînement final de l'architecture optimale sur CIFAR-10...")

valset_final = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

indices = list(range(len(full_trainset_final)))
np.random.shuffle(indices)
split = int(np.floor(0.1 * len(full_trainset_final)))
train_idx, val_idx = indices[split:], indices[:split]

train_dataset_final = Subset(full_trainset_final, train_idx)
val_dataset_final = Subset(valset_final, val_idx)

train_loader_final = DataLoader(train_dataset_final, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
val_loader_final = DataLoader(val_dataset_final, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

final_model = DynamicNet(best_sol_final, input_shape=(3, 32, 32)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

EPOCHS = 100
patience_limit = 10
patience_counter = 0
best_val_loss = float('inf')
best_weights = None

import copy

for epoch in range(EPOCHS):
    final_model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader_final:
        inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    scheduler.step()
    
    final_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader_final:
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
    val_loss = val_loss / len(val_loader_final)
    
    if (epoch + 1) % 10 == 0 or val_loss < best_val_loss:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader_final):.4f} | Acc: {100.*correct/total:.2f}% | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = copy.deepcopy(final_model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience_limit:
        print(f"Early stopping déclenché à l'itération {epoch+1}.")
        break

final_model.load_state_dict(best_weights)

final_model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        outputs = final_model(inputs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())

classes = ['avion', 'auto', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'navire', 'camion']
print("\nRésultats finaux sur le jeu de test CIFAR-10 :")
print(classification_report(all_targets, all_preds, target_names=classes))

report_str = classification_report(all_targets, all_preds, target_names=classes)
report_dict = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)

print("\nRésultats finaux sur le jeu de test CIFAR-10 :")
print(report_str)

import os
import json

os.makedirs("results", exist_ok=True)

torch.save(final_model.state_dict(), "results/best_model_weights.pth")

final_stats = {
    "search_stats": {
        "transformer_optimizer": stats_trans,
        "abc_optimizer": optim_stats_abc
    },
    "training_stats": {
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1,
        "early_stopping_triggered": patience_counter >= patience_limit
    },
    "classification_report": report_dict
}

with open("results/training_stats.json", "w", encoding="utf-8") as f:
    json.dump(final_stats, f, indent=4)

with open("results/training_logs.txt", "w", encoding="utf-8") as f:
    f.write("Résultats de l'entraînement final CIFAR-10\n")
    f.write("==========================================\n\n")
    f.write(f"Gain Transformer : {stats_trans['gain']:.2f}\n")
    f.write(f"Gain ABC : {optim_stats_abc['gain']:.2f}\n")
    f.write(f"Meilleure Loss de validation : {best_val_loss:.4f}\n")
    f.write(f"Époque d'arrêt : {epoch + 1}\n\n")
    f.write(report_str)