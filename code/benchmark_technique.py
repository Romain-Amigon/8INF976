import os
import time
import copy
import random
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg, MaxPool2dCfg, GlobalAvgPoolCfg, BatchNorm1dCfg, BatchNorm2dCfg, ResBlockCfg
from model import DynamicNet
from optimizer import SAOptimizer, GeneticOptimizer, ABCOptimizer, RLOptimizer, TransformerOptimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Optimisations Ampere (RTX 30XX) ---
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

BATCH_SIZE = 64
N_SAMPLES_TRAIN_IMG = 2000
N_SAMPLES_TEST_IMG = 500
N_STATS_RUNS = 5
ITERATIONS_OPTIM = 40

def get_dataset(task_type):
    if task_type == 'california_housing':
        data = fetch_california_housing()
        X, y = data.data, data.target
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1)), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1)), batch_size=BATCH_SIZE)
        return train_loader, test_loader, (8,), 1

    elif task_type == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=BATCH_SIZE)
        return train_loader, test_loader, (30,), 2

    elif 'fashion_mnist' in task_type:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        try:
            train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
            test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        except:
            return None, None, None, None

        train_loader = DataLoader(Subset(train_data, torch.randperm(len(train_data))[:N_SAMPLES_TRAIN_IMG]), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(Subset(test_data, torch.randperm(len(test_data))[:N_SAMPLES_TEST_IMG]), batch_size=BATCH_SIZE)
        return train_loader, test_loader, (1, 28, 28), 10

def get_initial_arch(task_type, input_shape, output_dim):
    layers = []
    if 'resblock' in task_type:
        layers.extend([
            Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU),
            BatchNorm2dCfg(num_features=16),
            ResBlockCfg(sub_layers=[
                Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU),
                BatchNorm2dCfg(num_features=16),
                Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=None) 
            ]),
            BatchNorm2dCfg(num_features=16),
            GlobalAvgPoolCfg(),
            LinearCfg(in_features=0, out_features=output_dim, activation=None)
        ])
    elif 'fashion_mnist' in task_type:
        layers.extend([
            Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU),
            BatchNorm2dCfg(num_features=16),
            MaxPool2dCfg(kernel_size=2, stride=2, padding=0),
            FlattenCfg(),
            LinearCfg(in_features=0, out_features=output_dim, activation=None)
        ])
    else: 
        layers.extend([
            FlattenCfg(),
            LinearCfg(in_features=0, out_features=32, activation=nn.ReLU),
            LinearCfg(in_features=0, out_features=output_dim, activation=None)
        ])
    return layers

def load_dataset_to_gpu(dataset):
    """Charge l'intégralité d'un sous-dataset en VRAM d'un seul coup."""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(loader))
    return X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

class BenchmarkWrapper:
    def __init__(self, optimizer_cls, task_type, **kwargs):
        self.optimizer_cls = optimizer_cls
        self.task_type = task_type
        self.kwargs = kwargs
        self.train_loader, self.test_loader, self.in_shape, self.out_dim = get_dataset(task_type)
        self.init_layers = get_initial_arch(task_type, self.in_shape, self.out_dim)
        
        base_dataset = self.train_loader.dataset
        train_size = int(0.8 * len(base_dataset))
        val_size = len(base_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(base_dataset, [train_size, val_size])
        
        # ---> PRÉCHARGEMENT MASSIF EN VRAM (Proxy ultra-rapide) <---
        self.X_train, self.y_train = load_dataset_to_gpu(train_subset)
        self.X_val, self.y_val = load_dataset_to_gpu(val_subset)

    def run(self, n_iterations):
        # On passe un dataset bidon juste pour l'initialisation des paramètres (out_features)
        dummy_loader = DataLoader(self.train_loader.dataset, batch_size=1)
        opt = self.optimizer_cls(layers=copy.deepcopy(self.init_layers), dataset=dummy_loader, **self.kwargs)
        
        def adaptive_evaluate(genome, train_epochs=5):
            opt.eval_count += 1 
            try:
                model = DynamicNet(genome, input_shape=self.in_shape).to(DEVICE)
                is_reg = ('regression' in self.task_type or 'california' in self.task_type)
                criterion = nn.MSELoss() if is_reg else nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                bs = BATCH_SIZE
                n = len(self.X_train)

                model.train()
                for epoch in range(train_epochs):
                    perm = torch.randperm(n, device=DEVICE)
                    for i in range(0, n, bs):
                        idx = perm[i:i+bs]
                        optimizer.zero_grad()
                        loss = criterion(model(self.X_train[idx]), self.y_train[idx])
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    pred = model(self.X_val)
                    if is_reg:
                        loss_val = criterion(pred, self.y_val).item()
                        return -loss_val
                    else:
                        _, predicted = torch.max(pred, 1)
                        correct = (predicted == self.y_val).sum().item()
                        return 100.0 * correct / max(1, len(self.y_val))

            except Exception:
                return -float('inf')

        opt.evaluate = adaptive_evaluate
        start_time = time.time()
        best_sol, optim_stats = opt.run(n_iterations)
        
        # ---> CORRECTION : Sauvegarde du compteur avant la suppression <---
        final_eval_count = opt.eval_count
        
        # ---> NETTOYAGE VRAM DE FIN DE RUN <---
        del self.X_train, self.y_train, self.X_val, self.y_val
        del opt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "score": optim_stats["best_score"],
            "search_time": time.time() - start_time,
            "gain": optim_stats["gain"],
            "best_iter": optim_stats["best_iter"],
            "depth_delta": len(best_sol) - len(self.init_layers),
            "evals": optim_stats.get("n_evaluations", final_eval_count) # On utilise la valeur sauvegardée
        }

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    tasks = ["california_housing", "breast_cancer", "fashion_mnist_simple"]
    
    optimizers = [
        ("Simulated Annealing", SAOptimizer, {"temp_init": 100, "cooling_rate": 0.7}),
        ("ABC Algorithm", ABCOptimizer, {"pop_size": 20, "limit": 5}), 
        ("Transformer", TransformerOptimizer, {"max_layers":50, "entropy_fct":"default"})
    ]

    print("\n" + "="*130)
    print(f"REAL DATASETS BENCHMARK (Runs: {N_STATS_RUNS} | Iterations: {ITERATIONS_OPTIM}) | DEVICE: {DEVICE}")
    print("="*130)

    results = []

    for task in tasks:
        print(f"\n>>> TASK: {task.upper()}")
        for opt_name, opt_cls, opt_params in optimizers:
            print(f"  > Running {opt_name}...", end="", flush=True)
            
            metrics = {"scores": [], "times": [], "gains": [], "depths": [], "iters": [], "evals": []}

            for i in range(N_STATS_RUNS):
                res = BenchmarkWrapper(opt_cls, task, **opt_params).run(ITERATIONS_OPTIM)
                if res["score"] > -float('inf'):
                    metrics["scores"].append(res["score"])
                    metrics["times"].append(res["search_time"])
                    metrics["gains"].append(res["gain"])
                    metrics["depths"].append(res["depth_delta"])
                    metrics["iters"].append(res["best_iter"])
                    metrics["evals"].append(res["evals"])
            
            if len(metrics["scores"]) > 0:
                results.append({
                    "task": task, "algo": opt_name,
                    "score_str": f"{np.mean(metrics['scores']):.2f} ± {np.std(metrics['scores'], ddof=1):.2f}",
                    "mean_score": np.mean(metrics['scores']),
                    "std_score": np.std(metrics['scores'], ddof=1),
                    "gain": np.mean(metrics["gains"]), "iter": np.mean(metrics["iters"]),
                    "depth": np.mean(metrics["depths"]), "time": np.mean(metrics["times"]),
                    "evals": np.mean(metrics["evals"])
                })
                print(f" Done in {np.sum(metrics['times']):.1f}s total.")
            else:
                print(" FAILED.")

    print("\n" + "="*130)
    header = f"{'TASK':<25} | {'ALGORITHM':<22} | {'PROXY SCORE (Avg±Std)':<20} | {'EVALS':<6} | {'GAIN':<8} | {'Δ DEPTH':<8} | {'TIME(s)':<8}"
    print(header)
    print("-" * 130)
    for r in results:
        print(f"{r['task']:<25} | {r['algo']:<22} | {r['score_str']:<20} | {r['evals']:<6.0f} | {r['gain']:<8.2f} | {r['depth']:<+8.1f} | {r['time']:<8.2f}")

    # =========================================================================
    # ENREGISTREMENT DES RÉSULTATS
    # =========================================================================
    
    summary_dict = {
        "runs": N_STATS_RUNS,
        "iterations": ITERATIONS_OPTIM,
        "results": results
    }
    with open("results/academic_benchmark_technique.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4)

    with open("results/academic_benchmark_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"REAL DATASETS BENCHMARK (Runs: {N_STATS_RUNS} | Iterations: {ITERATIONS_OPTIM})\n")
        f.write("="*130 + "\n")
        f.write(header + "\n")
        f.write("-" * 130 + "\n")
        for r in results:
            f.write(f"{r['task']:<25} | {r['algo']:<22} | {r['score_str']:<20} | {r['evals']:<6.0f} | {r['gain']:<8.2f} | {r['depth']:<+8.1f} | {r['time']:<8.2f}\n")
    
    print("\n--> Benchmark sauvegardé dans 'results/academic_benchmark_technique.json' et 'results/academic_benchmark_summary.txt'")